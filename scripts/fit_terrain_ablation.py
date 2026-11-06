"""Paired offline 10cm/2cm world-model pilot: identical initial weights and batches, no actor."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from scripts.fit_contact_world_model import evaluate
from lite3_supermimic.locomotion.contact_data import ContactDataset, check_splits, file_hash
from lite3_supermimic.locomotion.contact_model import ContactWorldModel, labels
from lite3_supermimic.supermimic import state as st
from lite3_supermimic.target import terrain as tr


class CheckedDataset(ContactDataset):
    def relief_at(self, ids, pose, device):
        rows=self.index[ids]
        for cid in rows[:,0].unique().tolist():
            field=self.fields[cid]
            if field is None: continue
            selected=pose[(rows[:,0]==cid).to(device)]
            points=torch.cat((selected[:,None,:2],tr.scan_world(selected[:,:2],selected[:,2])),1)
            x=(points[...,0]-field.origin[0])/field.res
            y=(points[...,1]-field.origin[1])/field.res
            if bool(((x<0)|(y<0)|(x>field.h.shape[1]-1.001)|(y>field.h.shape[0]-1.001)).any()):
                raise ValueError('Predicted scan left stored field: refusing silently clamped terrain')
        return super().relief_at(ids,pose,device)


def paired_loss(model, dataset, ids, device):
    b=dataset.batch(ids,device)
    s,pose=b['s0'][:,0],b['pose0'][:,0]
    scale=model.out_norm.var.sqrt()+1e-6
    dynamics,contact=0.,0.
    for j in range(dataset.horizon):
        relief=dataset.relief_at(ids,pose,device)
        s,delta,logits=model.predict(s,b['action'][:,j],b['prev'][:,j],b['context'][:,j],relief)
        pose=st.apply_pose_delta(pose,delta)
        err=torch.cat((s-b['s1'][:,j],delta-b['delta'][:,j]),-1)
        dynamics=dynamics+(err/scale).square().mean()
        contact=contact+F.binary_cross_entropy_with_logits(logits,labels(b['contact'][:,j]))
    return (dynamics+.2*contact)/dataset.horizon


def verify_pair(old,new):
    if not torch.equal(old.index,new.index): raise ValueError('Paired window indices differ')
    for om,nm in zip(old.manifests,new.manifests):
        change=nm['representation_change']
        for key in ('seed','checkpoint_sha256','stage','terrain','terrain_level','command','dt','num_envs','seconds'):
            if om['metadata'][key]!=nm['metadata'][key]: raise ValueError(f'Collection differs: {key}')
        src=Path(change['source_directory'])
        if file_hash(src/'manifest.json')!=change['source_manifest_sha256']: raise ValueError('Original manifest changed')
        for name,digest in change['source_chunks_sha256'].items():
            if file_hash(src/name)!=digest: raise ValueError('Original trajectory changed')
    for cid,(a,b) in enumerate(zip(old.chunks,new.chunks)):
        for key in a:
            if key not in ('s0','s1','relief') and not torch.equal(a[key],b[key]):
                raise ValueError(f'Paired physical data differs: {key}')
        for key in ('s0','s1'):
            if not torch.equal(a[key][...,1:],b[key][...,1:]): raise ValueError('Non-height state changed')
            pose_key='pose0' if key=='s0' else 'pose1'
            za=a[key][...,0]+tr.ground(a[pose_key][...,:2],old.fields[cid])
            zb=b[key][...,0]+tr.ground(b[pose_key][...,:2],new.fields[cid])
            if not torch.allclose(za,zb,atol=3e-6,rtol=0): raise ValueError('World base height changed during conversion')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='audit/contact_phase1/head_refine/contact_world_model.pt')
    p.add_argument('--old-root',default='audit/contact_phase1/data')
    p.add_argument('--fine-root',default='audit/terrain_2cm_pilot/data')
    p.add_argument('--output-dir',default='audit/terrain_2cm_pilot/fit')
    p.add_argument('--updates',type=int,default=500)
    p.add_argument('--batch',type=int,default=256)
    p.add_argument('--eval-windows',type=int,default=2048)
    p.add_argument('--eval-interval',type=int,default=100)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if min(a.updates,a.batch,a.eval_windows,a.eval_interval)<1: p.error('counts must be positive')
    out=Path(a.output_dir)
    if out.exists() and any(out.iterdir()): p.error('output-dir must be empty')
    saved=torch.load(a.source,map_location='cpu',weights_only=False)
    if saved.get('schema')!='lite3-contact-wm-offline-v1': p.error('Expected offline contact artifact')
    source_digest=file_hash(a.source)
    datasets={}
    for name,root in (('coarse',a.old_root),('fine',a.fine_root)):
        datasets[name]=[CheckedDataset([Path(root)/s/'up8_v10'],5) for s in ('train','validation','test')]
        check_splits(*datasets[name])
        for ds in datasets[name]:
            if any(m['metadata']['checkpoint_sha256']!=saved['source_sha256'] for m in ds.manifests):
                p.error('Model/data source mismatch')
    for old,new in zip(datasets['coarse'],datasets['fine']): verify_pair(old,new)
    gen=torch.Generator().manual_seed(1042)
    vi=torch.randperm(len(datasets['coarse'][1].index),generator=gen)[:a.eval_windows]
    ti=torch.randperm(len(datasets['coarse'][2].index),generator=gen)[:a.eval_windows]
    out.mkdir(parents=True,exist_ok=True)
    reports={}
    for name in ('coarse','fine'):
        torch.manual_seed(42)
        train,val,test=datasets[name]
        model=ContactWorldModel(tuple(saved['wm_hidden'])).to(a.device)
        model.load_state_dict(saved['world_model'])
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
        train_gen=torch.Generator().manual_seed(2042)
        initial=evaluate(model,val,vi,a.device)
        def score(report):
            g=report['step_5']['all']
            return g['normalized_mse']+.2*g['bce']
        best,best_it=score(initial),0
        best_weights=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
        before=evaluate(model,test,ti,a.device)
        for it in range(1,a.updates+1):
            ids=torch.randint(len(train.index),(a.batch,),generator=train_gen)
            loss=paired_loss(model,train,ids,a.device)
            if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite offline loss')
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            if it==1 or it%50==0: print(f'{name} {it}/{a.updates} loss={float(loss.detach()):.5g}; actor frozen',flush=True)
            if it%a.eval_interval==0 or it==a.updates:
                result=evaluate(model,val,vi,a.device)
                value=score(result)
                print(f'{name} validation score={value:.5g}',flush=True)
                if value<best:
                    best,best_it=value,it
                    best_weights=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
        model.load_state_dict(best_weights)
        for key,value in model.state_dict().items():
            if key.startswith(('in_norm.','out_norm.')) and not torch.equal(value.cpu(),saved['world_model'][key]):
                raise RuntimeError(f'Normalization unexpectedly changed: {key}')
        after=evaluate(model,test,ti,a.device)
        reports[name]=dict(selected_update=best_it,before_test=before,after_test=after)
        # Distinct schema prevents accidentally feeding a fine-reference model
        # into the old policy trainer or old contact-model evaluation workflow.
        torch.save(dict(schema='lite3-terrain-ablation-offline-v1',representation=name,
                        actor_updated=False,world_model=best_weights,wm_hidden=saved['wm_hidden'],
                        source_sha256=saved['source_sha256'],parent_model_sha256=source_digest,
                        args=vars(a),selected_update=best_it),out/f'{name}_world_model.pt')
        del optimizer,model
    if file_hash(a.source)!=source_digest: raise RuntimeError('Source model changed')
    summary=dict(actor_updated=False,policy_inputs_changed=False,source_model_unchanged=True,
                 scope='up8_v10 only; repeated seed-2 regression test, not fresh generalization',
                 selection='Each representation selected on validation MSE + 0.2 BCE, including update zero',
                 normalization='same frozen source buffers; base-height reference differs, compare physical metrics too',
                 updates_per_arm=a.updates,comparison={})
    for step in ('step_1','step_5'):
        summary['comparison'][step]={}
        for name in ('coarse','fine'):
            r=reports[name]['after_test'][step]['all']
            summary['comparison'][step][name]=dict(selected_update=reports[name]['selected_update'],
                **{k:v for k,v in r.items() if k!='contacts'},fr_shank_only=r['contacts']['shank_only_FR'])
    (out/'report.json').write_text(json.dumps(dict(summary=summary,arms=reports),indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    print(f'Saved {out}; offline artifacts are not actor checkpoints',flush=True)


if __name__=='__main__': main()
