"""Read-only campaign dashboard; serves telemetry and plot artifacts only."""

import argparse
import json
import mimetypes
import re
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


def read_events(directory):
    path = directory / "events.jsonl"
    events = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # The writer can be in the middle of appending the latest event.
                continue
    return events


def payload(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    events = read_events(directory)
    status = directory / "status.json"
    log = directory / "runner.log"
    contents = ""
    if log.exists():
        with log.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 24000))
            contents = stream.read().decode(errors="replace")
    links = [event for event in events if event["kind"] == "wandb_run"]
    return {"campaign": manifest["campaign_id"], "config": manifest["config"],
            "model_config": manifest["model_config"], "settings": manifest["settings"],
            "state": json.loads(status.read_text()) if status.exists() else {},
            "events": events, "wandb": links[0] if links else None,
            "log": re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", contents)}


FAVICON = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
<rect width="32" height="32" rx="7" fill="#101820"/>
<path d="M6 6v20h21" fill="none" stroke="#aebfcd" stroke-width="2"/>
<path d="M9 11l6 3 5 2 6 7v-7l-6-6-5-2-6-2z" fill="#66c9ff" opacity=".25"/>
<path d="M9 8l6 3 5 3 6 6" fill="none" stroke="#66c9ff" stroke-width="3" stroke-linecap="round"/>
</svg>'''


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PHM experiment monitor</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<style>
:root{color-scheme:dark;font:15px system-ui;background:#101820;color:#e5edf4}body{max-width:1400px;margin:24px auto;padding:0 20px}h1{font-size:28px}h2{font-size:19px}.panel{background:#192733;padding:18px;border-radius:10px;margin:16px 0}.cards{display:flex;gap:12px;flex-wrap:wrap}.card{flex:1;min-width:130px;background:#243847;padding:16px;border-radius:8px}.label,.hint{color:#aebfcd}.label{font-size:12px}.value{font-size:20px;margin-top:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:400px;overflow:auto;font-size:12px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:8px;border-bottom:1px solid #304555}th{color:#aebfcd}.numeric{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow:auto}a{color:#66c9ff;margin-right:20px}select{padding:7px;background:#243847;color:inherit;border:1px solid #587083;border-radius:5px}svg{width:100%;height:240px}img{width:100%;background:white;border-radius:8px}details{margin:14px 0}.best{color:#81e0a3}progress{width:100%;height:18px;margin:16px 0}.plot{margin-top:22px}
</style>
<h1 id="title">PHM experiment monitor</h1><p id="connection" class="hint">Connecting…</p>
<section class="panel"><div id="cards" class="cards"></div><progress id="progress" max="5" value="0"></progress><div id="wandb">W&B links appear when the first run starts.</div></section>
<section class="panel"><h2>Hyperparameter configuration</h2><div id="config"></div><details><summary>All experiment and model settings</summary><pre id="all-config"></pre></details></section>
<section class="panel"><h2>Epoch losses</h2><label>Fold / quantile <select id="series"></select></label><div id="chart"></div></section>
<section class="panel"><h2>Quantile metrics — held-out test lives</h2><p class="hint" id="metric-description"></p><p class="hint">Each fold table appears after all its quantile runs finish. The average table appears after all folds finish.</p><div id="tables"></div></section>
<section class="panel"><h2>Best fold plots</h2><p class="hint" id="best">Waiting for all folds. Selection uses the mean validation loss of the saved checkpoints across quantiles.</p><div id="plots"></div></section>
<section class="panel"><h2>Live pipeline log</h2><pre id="log"></pre></section>
<script>
const $=id=>document.getElementById(id);let current;
function node(tag,text){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n}
function number(v){return typeof v==='number'?Number(v.toPrecision(6)).toString():v==null?'—':String(v)}
function table(t){const wrap=node('div');wrap.className='scroll';const el=node('table'),header=node('tr');for(const c of [t.label||'Life',...t.columns])header.append(node('th',c));const head=node('thead');head.append(header);el.append(head);const body=node('tbody');t.index.forEach((label,i)=>{const row=node('tr');row.append(node('td',label));for(const value of t.values[i]){const cell=node('td',number(value));cell.className='numeric';row.append(cell)}body.append(row)});el.append(body);wrap.append(el);return wrap}
function chart(){const events=current.events.filter(e=>e.kind==='epoch'&&`${e.fold}/${e.tau}`===$('series').value);$('chart').replaceChildren();if(!events.length){$('chart').append(node('p','Loss curves appear after the first epoch.'));return}const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 900 240');const keys=['train_loss','val_loss','eval_val_loss'],colors=['#63ccff','#ffaa63','#a5e679'];const maxX=Math.max(1,...events.map(e=>e.epoch)),values=events.flatMap(e=>keys.map(k=>e[k])).filter(v=>typeof v==='number'&&Number.isFinite(v)),maxY=Math.max(1e-9,...values);keys.forEach((key,i)=>{const points=events.filter(e=>typeof e[key]==='number'&&Number.isFinite(e[key]));const line=document.createElementNS(ns,'polyline');line.setAttribute('points',points.map(e=>`${50+800*(e.epoch-1)/Math.max(1,maxX-1)},${190-165*e[key]/maxY}`).join(' '));line.setAttribute('fill','none');line.setAttribute('stroke',colors[i]);line.setAttribute('stroke-width','2');svg.append(line);const label=node('span',`${key}: ${number(points.at(-1)?.[key])}  `);label.style.color=colors[i];$('chart').append(label)});const label=document.createElementNS(ns,'text');label.setAttribute('x','50');label.setAttribute('y','225');label.setAttribute('fill','#aebfcd');label.textContent=`Epochs 1–${maxX} · loss range 0–${number(maxY)}`;svg.append(label);$('chart').append(svg)}
function render(data){current=data;const s=data.state,c=data.config,events=data.events,folds=events.filter(e=>e.kind==='fold_completed');$('title').textContent='PHM · '+data.campaign;document.title=data.campaign+' · PHM';$('cards').replaceChildren();for(const [label,value] of [['Status',s.status||'waiting'],['current run / total wandb run',`${new Set(events.filter(e=>e.kind==='wandb_run').map(e=>e.url)).size} / ${c.n_folds*c.quantiles.length}`],['Pipeline stage',s.stage||'—'],['Fold',`${s.fold||'—'} / ${c.n_folds}`],['Quantile',s.tau??'—'],['Epoch',`${s.epoch||'—'} / ${c.epochs}`],['Completed folds',folds.length]]){const card=node('div');card.className='card';const l=node('div',label);l.className='label';const v=node('div',value);v.className='value';card.append(l,v);$('cards').append(card)}$('progress').max=c.n_folds;$('progress').value=folds.length;
if(data.wandb){$('wandb').replaceChildren();for(const [label,url] of [['W&B project',data.wandb.project_url],['First run of the campaign',data.wandb.url]]){const parsed=new URL(url);if(parsed.protocol!=='https:'||parsed.hostname!=='wandb.ai')continue;const a=node('a',label+' ↗');a.href=url;a.target='_blank';a.rel='noopener';$('wandb').append(a)}}
const params=[['Backbone',data.settings.model_name],['Head',c.quantile_scale?'QuantileScaleHead':'QuantileHead'],['Folds',c.n_folds],['Epochs per quantile run',c.epochs],['Quantile levels',c.quantiles.join(', ')],['Optimizer learning rate',c.lr],['Batch size',c.batch_size],['Sequence length / stride',`${c.sequence_length} / ${c.stride}`],['Hidden size / layers / state size',`${data.model_config.d_model} / ${data.model_config.n_layers} / ${data.model_config.d_state}`],['Dropout',data.model_config.dropout],['Feature set',c.feature_type],['Approach',c.approach],['Loss / validation loss',`${c.loss} / ${c.eval_loss}`],['Total quantile training runs',c.n_folds*c.quantiles.length]];$('config').replaceChildren(table({label:'Parameter',columns:['Value'],index:params.map(p=>p[0]),values:params.map(p=>[p[1]])}));$('all-config').textContent=JSON.stringify({experiment:c,model:data.model_config,settings:data.settings},null,2);$('metric-description').textContent=`Metric: ${c.life_eval_loss.toUpperCase()} per life (padding zeros excluded).`;
$('tables').replaceChildren();const summary=events.filter(e=>e.kind==='summary').at(-1);for(const e of [...(summary?[summary]:[]),...folds]){$('tables').append(node('h3',e.kind==='summary'?'Average across all completed folds':`Fold ${e.fold} · mean checkpoint validation loss ${number(e.validation_loss)}`),table(e.table))}if(!folds.length)$('tables').append(node('p','No completed folds yet.'));
const chosen=events.filter(e=>e.kind==='best_fold_selected').at(-1);if(chosen)$('best').textContent=`Best fold: ${chosen.best_fold} · mean checkpoint validation loss ${number(chosen.validation_loss)}. Selected using validation data.`;
$('plots').replaceChildren();for(const e of events.filter(e=>e.kind==='plot')){const box=node('div');box.className='plot';box.append(node('h3',e.title));const img=node('img');img.src='/assets/'+encodeURIComponent(e.filename);img.alt=e.title;box.append(img);const link=node('a','Download PNG');link.href=img.src;link.download=e.filename;box.append(link);$('plots').append(box)}
const selected=$('series').value,groups=[...new Set(events.filter(e=>e.kind==='epoch').map(e=>`${e.fold}/${e.tau}`))];$('series').replaceChildren(...groups.map(v=>{const opt=node('option','Fold '+v.split('/')[0]+' · τ='+v.split('/')[1]);opt.value=v;return opt}));$('series').value=groups.includes(selected)?selected:(groups.at(-1)||'');chart();$('log').textContent=data.log;$('connection').textContent='Connected · refreshed '+new Date().toLocaleTimeString()+' · refreshes every 5 seconds'}
$('series').onchange=chart;async function refresh(){try{const response=await fetch('/api/state',{cache:'no-store'});if(!response.ok)throw Error(response.status);render(await response.json())}catch(e){$('connection').textContent='Connection unavailable · retrying in 5 seconds'}setTimeout(refresh,5000)}refresh();
</script></html>'''


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, directory, **kwargs):
        self.directory = directory
        super().__init__(*args, **kwargs)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path == "/":
            body, mime = HTML.encode(), "text/html; charset=utf-8"
        elif path == "/favicon.svg":
            body, mime = FAVICON.encode(), "image/svg+xml"
        elif path == "/api/state":
            body, mime = json.dumps(payload(self.directory), allow_nan=False).encode(), "application/json"
        elif path.startswith("/assets/"):
            root = (self.directory / "plots").resolve()
            target = (root / path[len("/assets/"):]).resolve()
            if not target.is_relative_to(root) or target.suffix not in (".png", ".pdf") or not target.is_file():
                self.send_error(404)
                return
            body, mime = target.read_bytes(), mimetypes.guess_type(target.name)[0]
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8878)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=args.directory))
    server.serve_forever()
