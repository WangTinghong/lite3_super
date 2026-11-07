"""One-step same-prefix action response: contact WM prediction versus MuJoCo."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def summarize_counterfactual(raw):
    import torch
    reference=raw['baseline']
    active=reference['active']
    common=active.clone()
    for item in raw.values():
        if not torch.equal(item['active'],active):
            raise ValueError('Active masks differ across cases')
        common &= ~item['done']
    if not bool(common.any()):
        raise ValueError('No common valid active transitions')
    output={}
    for name,item in raw.items():
        e=dict(item['metadata'])
        e['common_valid']=int(common.sum())
        e['predicted_fr_shank_only_probability']=float(item['prob'][common].mean())
        e['actual_fr_shank_only_fraction']=float(item['actual'][common].float().mean())
        e['fr_foot_contact_fraction']=float(item['foot'][common].float().mean())
        output[name]=e
    bp=output['baseline']['predicted_fr_shank_only_probability']
    ba=output['baseline']['actual_fr_shank_only_fraction']
    for e in output.values():
        e['predicted_risk_delta_vs_baseline']=e['predicted_fr_shank_only_probability']-bp
        e['actual_risk_delta_vs_baseline']=e['actual_fr_shank_only_fraction']-ba
    return output


def verify_same_prefix(raw, *, atol=1e-5):
    """Reject comparisons whose pre-intervention simulations are not matched."""
    import torch
    baseline=raw['baseline']
    failures=[]
    diagnostics={}
    for name,item in raw.items():
        if name=='baseline': continue
        if item['step']!=baseline['step']:
            failures.append(f'{name}: trigger step {item["step"]} != {baseline["step"]}')
        if not torch.equal(item['active'],baseline['active']):
            failures.append(f'{name}: active masks differ')
        field_diffs={}
        for field,expected in baseline['pre'].items():
            actual=item['pre'][field]
            if expected.shape!=actual.shape:
                failures.append(f'{name}.{field}: shape {tuple(actual.shape)} != {tuple(expected.shape)}')
                continue
            if expected.dtype==torch.bool:
                differences=int((actual!=expected).sum())
                field_diffs[field]=differences
                if differences: failures.append(f'{name}.{field}: {differences} boolean entries differ')
            else:
                delta=(actual-expected).abs()
                maximum=float(delta.max()) if delta.numel() else 0.
                field_diffs[field]=maximum
                if not torch.isfinite(delta).all() or maximum>atol:
                    failures.append(f'{name}.{field}: max abs difference {maximum:.6g} > {atol:g}')
        diagnostics[name]=field_diffs
    if failures:
        raise RuntimeError('Same-prefix states diverged; refusing causal comparison. '
                           + '; '.join(failures))
    return diagnostics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--contact-model',default='audit/contact_phase1/head_refine/contact_world_model.pt')
    p.add_argument('--output',default='audit/action_response/summary.json')
    p.add_argument('--seeds',type=int,nargs='+',default=[3,4,5])
    p.add_argument('--num-envs',type=int,default=64)
    p.add_argument('--max-seconds',type=float,default=20.)
    p.add_argument('--min-active',type=int,default=4)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--case',choices=('baseline','hipy_minus','hipy_plus','knee_minus','knee_plus'),help=argparse.SUPPRESS)
    p.add_argument('--raw-output',help=argparse.SUPPRESS)
    a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds) or any(s<3 for s in a.seeds): p.error('Use unique unused seeds >=3')
    if min(a.num_envs,a.max_seconds,a.min_active)<=0: p.error('Positive counts/seconds required')
    out=Path(a.output)
    if out.exists() and not a.worker: p.error('Output already exists')
    cases=(('baseline',0.,0.),('hipy_minus',-.1,0.),('hipy_plus',.1,0.),
           ('knee_minus',0.,-.1),('knee_plus',0.,.1))
    if not a.worker:
        import torch
        from scripts.contact_phase1 import child_environment
        from lite3_supermimic.locomotion.contact_data import file_hash
        result=dict(actor_updated=False,world_model_updated=False,one_step_only=True,
                    source_sha256=file_hash(a.checkpoint),model_sha256=file_hash(a.contact_model),
                    terrain='stairs_inv',terrain_level=8,command=[1.,0.,0.],cases={},
                    prefix_tolerance=1e-5)
        with tempfile.TemporaryDirectory(prefix='lite3_action_response_') as temp:
            for seed in a.seeds:
                seed_raw={}
                for name,_,_ in cases:
                    raw_path=Path(temp)/f'{seed}_{name}.pt'
                    command=[sys.executable,'-u',str(Path(__file__).resolve()),
                             '--checkpoint',a.checkpoint,'--contact-model',a.contact_model,
                             '--output',a.output,'--seeds',str(seed),'--num-envs',str(a.num_envs),
                             '--max-seconds',str(a.max_seconds),'--min-active',str(a.min_active),
                             '--device',a.device,'--worker','--case',name,
                             '--raw-output',str(raw_path)]
                    worker_env=child_environment()
                    worker_env['PYTHONHASHSEED']=str(seed)
                    subprocess.run(command,env=worker_env,check=True)
                    seed_raw[name]=torch.load(raw_path,map_location='cpu',weights_only=False)
                differences=verify_same_prefix(seed_raw)
                result['cases'][str(seed)]=summarize_counterfactual(seed_raw)
                result.setdefault('prefix_max_abs_difference',{})[str(seed)]=differences
                print(f'seed {seed}: same-prefix check passed',flush=True)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
        print(f'Saved {out}. One-step response only; no actor optimization.',flush=True)
        return
    if a.case is None or a.raw_output is None or len(a.seeds)!=1:
        p.error('--worker requires one --case, one --seeds value, and --raw-output')
    import random
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from lite3_supermimic.locomotion.env import make_env_cfg
    from lite3_supermimic.locomotion.core import validate_checkpoint
    from lite3_supermimic.locomotion.contact_data import file_hash
    from lite3_supermimic.locomotion.contact_model import ContactWorldModel
    from lite3_supermimic.locomotion.metrics import read_contacts,read_shank_contacts
    from lite3_supermimic.locomotion.stair_probe import front_leg_probe
    from lite3_supermimic.supermimic import state as st
    from lite3_supermimic.supermimic.train import Actor
    from lite3_supermimic.target import terrain as tr
    actor_ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    validate_checkpoint(actor_ckpt)
    wm_saved=torch.load(a.contact_model,map_location='cpu',weights_only=False)
    if wm_saved.get('schema')!='lite3-contact-wm-offline-v1' or wm_saved['source_sha256']!=file_hash(a.checkpoint):
        p.error('Contact model does not match frozen policy source')
    wm=ContactWorldModel(tuple(wm_saved['wm_hidden'])).to(a.device)
    wm.load_state_dict(wm_saved['world_model']); wm.eval()
    for seed in a.seeds:
        for name,hipy,knee in cases:
            if name!=a.case: continue
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            cfg=make_env_cfg('terrain',play=True,harder_stairs=bool(actor_ckpt['cfg'].get('harder_stairs',False)))
            cfg.seed=seed; cfg.scene.num_envs=a.num_envs; cfg.auto_reset=False
            command_cfg=cfg.commands['velocity']; command_cfg.fixed_level=8; command_cfg.fixed_command=(1.,0.,0.)
            generator=cfg.scene.terrain.terrain_generator
            tile=generator.sub_terrains['stairs_inv']; tile.proportion=1.
            generator.sub_terrains={'stairs_inv':tile}
            print(f'seed {seed}, {name}: initializing shared-prefix MuJoCo run',flush=True)
            env=ManagerBasedRlEnv(cfg,device=a.device)
            try:
                env.common_step_counter=command_cfg.curriculum_steps
                obs,_=env.reset()
                actor=Actor(obs['actor'].shape[-1],tuple(actor_ckpt['cfg']['actor_hidden']),12).to(env.device)
                actor.load_state_dict(actor_ckpt['actor']); actor.eval()
                term=env.command_manager.get_term('velocity')
                reader=st.ContextReader(env)
                base_id=int(reader.base_body)
                reader.base_ipos0=torch.as_tensor(env.sim.mj_model.body_ipos[base_id].copy(),
                    dtype=torch.float32,device=env.device).expand(env.num_envs,3)
                scale=st.robot_consts(env).action_scale
                prev=torch.zeros(env.num_envs,12,device=env.device)
                event=None
                with torch.inference_mode():
                    for step in range(max(1,int(a.max_seconds/env.step_dt))):
                        s0,p0=st.robot_state(env,term.height_field),st.planar_pose(env)
                        cmd=term.command.clone()
                        action=actor(obs['actor'])
                        foot0,_=read_contacts(env.scene['feet_ground_contact'])
                        context=reader.read()
                        _,gate=front_leg_probe(action,s0,p0,cmd,term.height_field,
                                                foot0,scale,-.1,0.)
                        active=(gate>.01)
                        if int(active.sum())>=a.min_active:
                            adjusted,_=front_leg_probe(action,s0,p0,cmd,term.height_field,
                                                       foot0,scale,hipy,knee)
                            # In baseline, the probe function returns original actions.
                            pre=dict(state=s0,pose=p0,action=action,previous_action=prev,
                                     context=context,foot_contact=foot0,command=cmd,gate=gate)
                            pre={key:value.detach().cpu().clone() for key,value in pre.items()}
                            relief=tr.relief(p0[:,:2],p0[:,2],term.height_field)
                            _,_,logits=wm.predict(s0,adjusted,prev,context,relief)
                            obs,_,terminated,truncated,_=env.step(adjusted)
                            feet1,_=read_contacts(env.scene['feet_ground_contact'])
                            shanks1,_=read_shank_contacts(env.scene['shanks_ground_contact'])
                            shank_only=shanks1[:,0] & ~feet1[:,0]
                            raw=dict(
                                step=step,pre=pre,
                                active=active.detach().cpu().clone(),
                                done=(terminated|truncated).detach().cpu().clone(),
                                prob=logits[:,8].sigmoid().detach().cpu().clone(),
                                actual=shank_only.detach().cpu().clone(),
                                foot=feet1[:,0].detach().cpu().clone(),
                                metadata=dict(step=step,active=int(active.sum()),
                                    falls=int(terminated.sum()),
                                    action_target_hipy_delta_rad=hipy,action_target_knee_delta_rad=knee,
                                    mean_gate=float(gate[active].mean()),
                                    fr_hipy_action_delta=float((adjusted[active,4]-action[active,4]).mean()),
                                    fr_knee_action_delta=float((adjusted[active,5]-action[active,5]).mean())))
                            event=True
                            break
                        obs,_,terminated,truncated,_=env.step(action)
                        done=terminated|truncated
                        prev=torch.where(done[:,None],torch.zeros_like(action),action)
                        if done.any(): obs,_=env.reset(env_ids=done.nonzero().flatten())
                if event is None:
                    raise RuntimeError(f'No common stair approach with {a.min_active} active envs in seed {seed}; extend --max-seconds or lower --min-active')
                print(f'seed {seed}, {name}: same-prefix intervention at step {step}; active={int(active.sum())}',flush=True)
            finally:
                env.close()
        torch.save(raw,a.raw_output)


if __name__=='__main__': main()
