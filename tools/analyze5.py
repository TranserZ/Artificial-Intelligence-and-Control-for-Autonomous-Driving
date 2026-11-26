import json, glob, os, sys, collections
rows=[]
for f in sorted(glob.glob('out/h5_*/log.json')):
    d=json.load(open(f)); st=d['steps']; last=st[-1]
    odo=max((r.get('odo',0) for r in st), default=0)
    for r in reversed(st):
        if 'odo' in r: odo=r['odo']; edge=r.get('edge'); v=r.get('v'); break
    kind=('col' if last.get('collided') else 'red' if last.get('red_violation') else 'wrong' if last.get('wrong_lane')
          else 'ok' if last.get('arrived') and not last.get('missed_stops') else 'miss' if last.get('arrived') else 'to')
    sh=sum(1 for r in st if r.get('shield_active'))/len(st)
    rows.append(dict(seed=d['seed'],kind=kind,odo=round(odo),edge=edge,v=v,steps=len(st),stops=last.get('stops_done'),missed=last.get('missed_stops'),shield=sh))
c=collections.Counter(r['kind'] for r in rows); n=len(rows)
print(n, {k:(v, round(100*v/n,1)) for k,v in c.items()}, 'shield mean %.3f'%(sum(r['shield'] for r in rows)/n))
for r in rows:
    if r['kind']!='ok': print(r['seed'],r['kind'],r['odo'],r['edge'],r['v'],'stops',r['stops'],'missed',r['missed'],'t',r['steps']//2)
if len(sys.argv)>1:
    fails=[[r['odo'],r['kind']] for r in rows if r['kind'] in('col','red','to')]
    open(sys.argv[1],'w').write('window.FAILS='+json.dumps(fails,separators=(',',':'))+';')
