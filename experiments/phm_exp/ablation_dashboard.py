"""Read-only PHM weight-ablation dashboard and self-contained offline exporter."""
import argparse
import base64
import csv
import json
import math
import re
import threading
import time
from datetime import datetime, timezone
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

FAVICON = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#101820"/><path d="M6 6v20h21" fill="none" stroke="#aebfcd" stroke-width="2"/><path d="M9 11l6 3 5 2 6 7v-7l-6-6-5-2-6-2z" fill="#66c9ff" opacity=".25"/><path d="M9 8l6 3 5 3 6 6" fill="none" stroke="#66c9ff" stroke-width="3" stroke-linecap="round"/></svg>'


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def events(directory):
    records = []
    path = directory / 'events.jsonl'
    if path.exists():
        with path.open() as stream:
            for line in stream:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    # Retain lifecycle events and a bounded recent epoch history per work unit.
    epochs = {}
    other = []
    for record in records:
        if record.get('kind') == 'epoch':
            key = (record.get('setting'), record.get('fold'), record.get('tau'))
            epochs.setdefault(key, []).append(record)
        else:
            other.append(record)
    return other + [event for rows in epochs.values() for event in rows[-1000:]]


def csv_rows(path):
    if not path.exists():
        return []
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if key == 'setting':
                continue
            try:
                number = float(value)
                row[key] = number if math.isfinite(number) else None
            except (ValueError, TypeError):
                row[key] = None if value == '' else value
    return rows


def payload(directory):
    manifest = read_json(directory / 'manifest.json', {})
    status = read_json(directory / 'status.json', {'status': 'waiting'})
    heartbeat = read_json(directory / 'heartbeat.json', {})
    observed = status.get('status', 'waiting')
    if observed == 'running' and time.time() - heartbeat.get('unix_time', 0) > 60:
        observed = 'stale / disconnected (last recorded state: running)'
    results = directory / 'ablation_results'
    data = {'manifest': manifest, 'state': status, 'observed_status': observed,
            'events': events(directory), 'tables': {},
            'plots': sorted(p.name for p in results.glob('*.png')), 'log': ''}
    for name in ('fold_rmse', 'summary_rmse', 'fold_pinball', 'summary_pinball', 'fold_intervals', 'summary_intervals'):
        data['tables'][name] = csv_rows(results / (name + '.csv'))
    path = directory / 'runner.log'
    if path.exists():
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 24000))
            data['log'] = re.sub(r'\x1b\[[0-9;?]*[a-zA-Z]', '', stream.read().decode(errors='replace'))
    return data


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PHM · Window weight ablation</title><link rel="icon" type="image/svg+xml" href="/favicon.svg"><style>
:root{font:15px system-ui;color-scheme:dark;background:#101820;color:#e5edf4}body{max-width:1400px;margin:28px auto;padding:0 22px}h1{font-size:28px}h2{font-size:20px}.panel{background:#192733;padding:20px;border-radius:12px;margin:18px 0}.cards{display:flex;flex-wrap:wrap;gap:12px}.card{flex:1;min-width:150px;background:#243847;padding:16px;border-radius:8px}.label,.hint{color:#aebfcd}.value{font-size:21px;margin-top:8px}a{color:#66c9ff}select{background:#243847;color:inherit;padding:8px;border-radius:6px;margin:8px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:9px;border-bottom:1px solid #304555;text-align:right;font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:400px;overflow:auto}progress{width:100%;height:18px;margin-top:15px}svg{width:100%;height:260px}img{width:100%;border-radius:8px;background:white}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px}button{background:#243847;color:inherit;padding:9px;border:1px solid #587083;border-radius:6px}
</style></head><body><h1 id="title">Window weight ablation</h1><p id="connection" class="hint">Connecting…</p><p><a id="download" href="/snapshot.html" download>Download offline HTML</a></p>
<section class="panel"><div id="cards" class="cards"></div><progress id="progress"></progress><p id="update" class="hint"></p><div id="wandb"></div></section>
<section class="panel"><h2>Effective configuration</h2><div id="config"></div><details><summary>Recorded command, configuration and provenance</summary><pre id="manifest"></pre></details></section>
<section class="panel"><h2>Epoch losses</h2><label>Weight setting / fold <select id="series"></select></label><div id="chart"></div><p class="hint">Training loss uses sampled quantiles; validation is at τ=0.5. Checkpoints minimize validation loss. Kernel learning rate and optimizer learning rate are distinct settings.</p></section>
<section class="panel"><h2>Held-out results</h2><p class="hint">Unweighted test metrics: average valid samples within each life, then average lives equally; fold summaries report mean ± sample SD, not confidence intervals. Constant = RUL 500; decreasing = RUL below 500; zero padding excluded. Partial summaries use only evaluated folds. No weight setting is selected from test results.</p><label>Table <select id="table"><option value="summary_rmse">Median RMSE by weight · lower is better (RUL units)</option><option value="summary_pinball">Pinball by weight and quantile · lower is better (RUL units)</option><option value="summary_intervals">80% intervals by weight · coverage target 0.8; width in RUL units</option><option value="fold_rmse">RMSE per evaluated fold</option><option value="fold_pinball">Pinball per evaluated fold and quantile</option><option value="fold_intervals">80% intervals per evaluated fold</option></select></label><div id="results"></div><p id="counts" class="hint"></p><a id="csv" href="#" download>Download selected CSV</a></section>
<section class="panel"><h2>Weight comparisons</h2><p class="hint">Points represent evaluated weight settings; segments only connect those points. Error bars are fold SD. Interval width must be read alongside coverage and crossing rate; zero crossing alone does not establish useful uncertainty.</p><label>Region <select id="region"><option>overall</option><option>constant</option><option>decreasing</option></select></label><div id="plots" class="grid"></div></section>
<section class="panel"><h2>Pipeline log</h2><pre id="log"></pre></section><script>
const $=id=>document.getElementById(id),saved=document.getElementById('saved-ablation'),offline=saved?JSON.parse(saved.textContent):null;let current;
function node(tag,text){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;return el}
function number(v){return typeof v==='number'&&Number.isFinite(v)?Number(v.toPrecision(6)).toString():v==null?'—':String(v)}
function table(rows){const wrap=node('div');wrap.className='scroll';if(!rows.length){wrap.append(node('p','Waiting for evaluated folds.'));return wrap}const t=node('table'),keys=Object.keys(rows[0]),head=node('tr');keys.forEach(k=>head.append(node('th',k.replaceAll('_',' '))));t.append(head);for(const row of rows){const tr=node('tr');keys.forEach(k=>tr.append(node('td',number(row[k]))));t.append(tr)}wrap.append(t);return wrap}
function results(){const key=$('table').value,rows=current.tables[key];$('results').replaceChildren(table(rows));$('counts').textContent=current.tables.fold_rmse.length+' evaluated folds; summaries are partial until every planned fold is evaluated.';const csv=[rows.length?Object.keys(rows[0]).join(','):'',...rows.map(r=>Object.values(r).map(v=>v==null?'':JSON.stringify(v)).join(','))].join('\n');$('csv').href=offline?'data:text/csv;charset=utf-8,'+encodeURIComponent(csv):'/results/'+key+'.csv';$('csv').download=key+'.csv'}
function chart(){const records=current.events.filter(e=>e.kind==='epoch'&&`${e.setting}/${e.fold}`===$('series').value);$('chart').replaceChildren();if(!records.length){$('chart').append(node('p','Waiting for the first completed epoch.'));return}const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 900 260');const keys=['train_loss','val_loss','eval_val_loss'],colors=['#66c9ff','#ffaa63','#a5e679'];const finite=records.flatMap(e=>keys.map(k=>e[k])).filter(v=>typeof v==='number'&&Number.isFinite(v)),maxY=Math.max(1e-9,...finite),maxX=Math.max(1,...records.map(e=>e.epoch));for(let i=0;i<keys.length;i++){const k=keys[i],values=records.filter(e=>typeof e[k]==='number'&&Number.isFinite(e[k]));const line=document.createElementNS(ns,'polyline');line.setAttribute('points',values.map(e=>`${55+800*(e.epoch-1)/Math.max(1,maxX-1)},${200-165*e[k]/maxY}`).join(' '));line.setAttribute('fill','none');line.setAttribute('stroke',colors[i]);line.setAttribute('stroke-width','2');svg.append(line);const label=node('span',k+' '+number(values.at(-1)?.[k])+'  ');label.style.color=colors[i];$('chart').append(label)}const label=document.createElementNS(ns,'text');label.setAttribute('x','55');label.setAttribute('y','240');label.setAttribute('fill','#aebfcd');label.textContent='Epochs 1–'+maxX+' · loss range 0–'+number(maxY);svg.append(label);$('chart').append(svg)}
function plots(){const region=$('region').value;$('plots').replaceChildren();for(const name of current.plots.filter(p=>p.startsWith(region+'_'))){const box=node('div'),caption=node('p',name.replaceAll('_',' ').replace('.png',''));const img=node('img');img.alt=caption.textContent;img.src=current.plot_assets?.[name]||('/results/'+encodeURIComponent(name));const a=node('a','Download PNG');a.href=img.src;a.download=name;box.append(caption,img,a);$('plots').append(box)}if(!$('plots').children.length)$('plots').append(node('p','Comparison plots appear after a fold is evaluated.'))}
function render(data){current=data;const m=data.manifest,s=data.state,e=data.events,c=m.config||{},mc=m.model_config||{},done=new Set(e.filter(x=>x.kind==='ablation_fold_evaluated').map(x=>x.setting+'/'+x.fold)).size,trained=new Set(e.filter(x=>x.kind==='fold_completed').map(x=>x.setting+'/'+x.fold)).size;document.title=m.campaign_id+' · PHM weight ablation';$('title').textContent='PHM · '+m.campaign_id;$('cards').replaceChildren();const elapsed=Math.max(0,((offline?Date.parse(data.snapshot_time):Date.now())-Date.parse(m.created_at))/1000),duration=(s.status==='completed'||s.status==='failed'||s.status==='interrupted')?Math.max(0,(Date.parse(s.time)-Date.parse(m.created_at))/1000):elapsed;for(const [label,value] of [['Status',data.observed_status],['Weight setting',s.setting||'—'],['Stage',s.stage||'waiting'],['Fold',`${s.fold||'—'} / ${c.n_folds}`],['Epoch',`${s.epoch||'—'} / ${c.epochs}`],['Trained / planned',`${trained} / ${m.planned_work_units}`],['Evaluated / planned',`${done} / ${m.planned_work_units}`],['Elapsed',Math.floor(duration/3600)+'h '+Math.floor(duration%3600/60)+'m']]){const card=node('div');card.className='card';const l=node('div',label);l.className='label';const v=node('div',value);v.className='value';card.append(l,v);$('cards').append(card)}$('progress').max=m.planned_work_units;$('progress').value=done;$('update').textContent='Last recorded update: '+(s.time||'waiting')+' · '+(done===m.planned_work_units&&s.status==='completed'?'Final results':'Partial results');
$('wandb').replaceChildren();const links=e.filter(x=>x.kind==='wandb_run');if(!m.wandb_enabled)$('wandb').textContent='W&B disabled; all telemetry and results are recorded locally.';else if(!links.length)$('wandb').textContent='W&B enabled · waiting for the first run.';for(const [i,run] of links.entries()){for(const [title,url] of [...(i===0?[['W&B project',run.project_url]]:[]),[`${run.setting} / fold ${run.fold}${i===0?' · first run':''}`,run.url]]){try{if(new URL(url).protocol!=='https:')continue;const a=node('a',title+' ↗ ');a.href=url;a.target='_blank';a.rel='noopener';$('wandb').append(a)}catch{}}}
const params={Model:m.settings?.model_name,Head:c.quantile_scale?'QuantileScaleHead':'QuantileHead','tau feature / multiplier':`${mc.tau_feat} / ${mc.tau_mult}`,'Weight ratios':m.ratios?.join(', '),'Training quantile': 'Sampled from '+c.quantile_dist+' '+JSON.stringify(c.bounds),'Evaluation quantiles':m.evaluation_quantiles?.join(', '),'Seed':m.seed,'Folds / epochs':`${c.n_folds} / ${c.epochs}`,'Optimizer LR / weight decay':`${c.lr} / ${c.weight_decay}`,'Kernel LR (model config)':mc.lr,'Hidden size / layers / state':`${mc.d_model} / ${mc.n_layers} / ${mc.d_state}`,'Dropout':mc.dropout,'Batch / sequence / stride':`${c.batch_size} / ${c.sequence_length} / ${c.stride}`,'Preprocessing / features':`${c.transformer_type} / ${c.feature_type}`,'Loss / validation loss':`${c.loss} / ${c.eval_loss}`,'RUL clipping / normalization':`${c.max_rul} / ${c.normalize_rul}`};$('config').replaceChildren(table(Object.entries(params).map(([Parameter,Value])=>({Parameter,Value}))));$('manifest').textContent=JSON.stringify(m,null,2);const selected=$('series').value,groups=[...new Set(e.filter(x=>x.kind==='epoch').map(x=>`${x.setting}/${x.fold}`))];$('series').replaceChildren(...groups.map(v=>{const o=node('option',v);o.value=v;return o}));$('series').value=groups.includes(selected)?selected:(groups.at(-1)||'');chart();results();plots();$('log').textContent=data.log;$('connection').textContent=offline?'Offline snapshot · captured '+data.snapshot_time+' · '+data.observed_status:'Connected · refreshed '+new Date().toLocaleTimeString()+' · every 5 seconds';$('download').hidden=Boolean(offline)}
$('series').onchange=chart;$('table').onchange=results;$('region').onchange=plots;async function refresh(){try{const r=await fetch('/api/state',{cache:'no-store'});if(!r.ok)throw Error(r.status);render(await r.json())}catch{$('connection').textContent='Refresh failed · showing last successful data · retrying'}setTimeout(refresh,5000)}if(offline)render(offline);else refresh();
</script></body></html>'''


def snapshot_html(directory):
    data = payload(directory)
    data['snapshot_time'] = datetime.now(timezone.utc).isoformat()
    data['plot_assets'] = {}
    for name in data['plots']:
        data['plot_assets'][name] = 'data:image/png;base64,' + base64.b64encode((directory / 'ablation_results' / name).read_bytes()).decode()
    serialized = json.dumps(data, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    icon = 'data:image/svg+xml;base64,' + base64.b64encode(FAVICON.encode()).decode()
    return HTML.replace('href="/favicon.svg"', f'href="{icon}"').replace('<script>','<script id="saved-ablation" type="application/json">'+serialized+'</script><script>',1)


def export_snapshot(directory, target=None):
    target = target or directory / 'dashboard_snapshot.html'
    temporary = target.with_suffix('.html.tmp')
    temporary.write_text(snapshot_html(directory), encoding='utf-8')
    temporary.replace(target)
    return target


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, directory, **kwargs):
        self.directory = directory
        super().__init__(*args, **kwargs)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        download = None
        if path == '/':
            body, mime = HTML.encode(), 'text/html; charset=utf-8'
        elif path == '/api/state':
            body, mime = json.dumps(payload(self.directory), allow_nan=False).encode(), 'application/json'
        elif path == '/favicon.svg':
            body, mime = FAVICON.encode(), 'image/svg+xml'
        elif path == '/snapshot.html':
            target = self.directory / 'dashboard_snapshot.html'
            if not target.exists():
                self.send_error(503, 'Snapshot is being prepared'); return
            body, mime = target.read_bytes(), 'text/html; charset=utf-8'
            download = 'phm-weight-ablation-' + self.directory.name + '.html'
        elif path.startswith('/results/'):
            root = (self.directory / 'ablation_results').resolve()
            target = (root / path[len('/results/'):]).resolve()
            if not target.is_relative_to(root) or target.suffix not in ('.png', '.csv') or not target.is_file():
                self.send_error(404); return
            body = target.read_bytes()
            mime = 'image/png' if target.suffix == '.png' else 'text/csv'
        else:
            self.send_error(404); return
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if download:
            self.send_header('Content-Disposition', f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(body)


def serve(directory, port):
    # Export independently of requests; never launch training or plots on refresh.
    server = ThreadingHTTPServer(('127.0.0.1', port), partial(Handler, directory=directory))
    stop = threading.Event()
    def exporter():
        while not stop.is_set():
            try:
                export_snapshot(directory)
            except Exception as error:
                print(f'Snapshot export failed; retaining previous file: {error}', flush=True)
            stop.wait(5)
    thread = threading.Thread(target=exporter, daemon=True)
    thread.start()
    try:
        server.serve_forever()
    finally:
        stop.set(); server.server_close(); thread.join(timeout=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8879)
    parser.add_argument('--export', type=Path)
    args = parser.parse_args()
    if args.export:
        print(export_snapshot(args.directory, args.export))
    else:
        serve(args.directory, args.port)
