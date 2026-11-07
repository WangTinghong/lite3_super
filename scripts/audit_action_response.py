"""One-step same-prefix action response: contact WM prediction versus MuJoCo."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def summarize_counterfactual(raw, matched=None, min_active=1):
    import torch
    reference=raw['baseline']
    active=reference['active']
    common=active.clone() if matched is None else (active & matched)
    for item in raw.values():
        if not torch.equal(item['active'][common],active[common]):
            raise ValueError('Active masks differ across matched cases')
        common &= ~item['done']
    if int(common.sum())<min_active:
        raise ValueError(f'No common valid active transitions: {int(common.sum())} < {min_active}')
    if not bool((reference['actual'] & common).any()):
        raise ValueError('Selected baseline shank event is absent from common valid transitions')
    output={}
    baseline_events=int((reference['actual'] & common).sum())
    for name,item in raw.items():
        e=dict(item['metadata'])
        e['common_valid']=int(common.sum())
        e['common_baseline_shank_events']=baseline_events
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


PREFIX_ABS_TOLERANCES={
    'state':1e-3,
    'pose':1e-4,
    'action':1e-4,
    'previous_action':1e-4,
    'context':1e-4,
    'command':1e-6,
    'gate':1e-4,
}


def verify_same_prefix(raw):
    """Match individual worlds whose pre-intervention states stay within bounds."""
    import torch
    baseline=raw['baseline']
    n=len(baseline['active'])
    matched=torch.ones(n,dtype=torch.bool)
    diagnostics={}
    for name,item in raw.items():
        if name=='baseline': continue
        if item['step']!=baseline['step']:
            raise RuntimeError(f'{name}: trigger step {item["step"]} != baseline {baseline["step"]}')
        if item['active'].shape!=baseline['active'].shape:
            raise RuntimeError(f'{name}: active mask shape differs')
        case_match=item['active']==baseline['active']
        field_diffs={}
        dropped_by_field={}
        for field,expected in baseline['pre'].items():
            actual=item['pre'][field]
            if expected.shape!=actual.shape:
                raise RuntimeError(f'{name}.{field}: shape {tuple(actual.shape)} != {tuple(expected.shape)}')
            if expected.shape[0]!=n:
                raise RuntimeError(f'{name}.{field}: missing environment dimension')
            if expected.dtype==torch.bool:
                per_env_equal=(actual==expected).reshape(n,-1).all(1)
                field_diffs[field]=int((~per_env_equal).sum())
            else:
                delta=(actual-expected).abs().reshape(n,-1).amax(1)
                field_diffs[field]=float(delta.max())
                tolerance=PREFIX_ABS_TOLERANCES[field]
                per_env_equal=torch.isfinite(delta) & (delta<=tolerance)
            dropped_by_field[field]=int((~per_env_equal & baseline['active']).sum())
            case_match &= per_env_equal
        matched &= case_match
        diagnostics[name]=dict(matched_envs=int(case_match.sum()),
                               matched_active=int((case_match & baseline['active']).sum()),
                               maximum_difference=field_diffs,
                               dropped_active_by_field=dropped_by_field)
    return matched,dict(total_envs=n,baseline_active=int(baseline['active'].sum()),
                        matched_envs=int(matched.sum()),matched_active=int((matched & baseline['active']).sum()),
                        dropped_active=int((~matched & baseline['active']).sum()),
                        per_case=diagnostics)


def verify_null_response(raw, matched=None, *, probability_atol=0.01):
    """A repeated baseline must give the same discrete next-step contact result."""
    import torch
    reference=raw['baseline']
    repeated=raw['baseline_repeat']
    active=reference['active'] if matched is None else (reference['active'] & matched)
    if not bool(active.any()):
        raise RuntimeError('No matched active environments for repeated baseline')
    failures=[]
    for field in ('done','actual','foot'):
        differing=int((reference[field][active]!=repeated[field][active]).sum())
        if differing:
            failures.append(f'{field}: {differing} active outcomes differ')
    probability_drift=float((reference['prob'][active]-repeated['prob'][active]).abs().max())
    if probability_drift>probability_atol:
        failures.append(f'predicted probability max drift {probability_drift:.6g} > {probability_atol:g}')
    if failures:
        raise RuntimeError('Repeated baseline one-step response differs; causal comparison inconclusive. '
                           + '; '.join(failures))
    return dict(active=int(active.sum()),max_probability_drift=probability_drift,
                discrete_contact_outcomes_identical=True)


def baseline_contact_event(active, shank_only, min_active):
    """Select an informative baseline step, before comparing counterfactuals."""
    return int(active.sum())>=min_active and bool((active & shank_only).any())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--contact-model',default='audit/contact_phase1/head_refine/contact_world_model.pt')
    p.add_argument('--output',default='audit/action_response/summary.json')
    p.add_argument('--seeds',type=int,nargs='+',default=[3,4,5])
    p.add_argument('--num-envs',type=int,default=64)
    p.add_argument('--max-seconds',type=float,default=40.)
    p.add_argument('--min-active',type=int,default=4)
    p.add_argument('--delta',type=float,default=0.1,
                   help='Absolute FR HipY/Knee intervention in radians (0 < delta <= 0.15)')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--case',choices=('baseline','baseline_repeat','hipy_minus','hipy_plus','knee_minus','knee_plus'),help=argparse.SUPPRESS)
    p.add_argument('--raw-output',help=argparse.SUPPRESS)
    p.add_argument('--trigger-step',type=int,default=-1,help=argparse.SUPPRESS)
    a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds) or any(s<3 for s in a.seeds): p.error('Use unique unused seeds >=3')
    if min(a.num_envs,a.max_seconds,a.min_active)<=0: p.error('Positive counts/seconds required')
    if not 0 < a.delta <= 0.15: p.error('--delta must be in (0, 0.15] rad')
    out=Path(a.output)
    if out.exists() and not a.worker: p.error('Output already exists')
    cases=(('baseline',0.,0.),('baseline_repeat',0.,0.),('hipy_minus',-a.delta,0.),('hipy_plus',a.delta,0.),
           ('knee_minus',0.,-a.delta),('knee_plus',0.,a.delta))
    if not a.worker:
        import torch
        from scripts.contact_phase1 import child_environment
        from lite3_supermimic.locomotion.contact_data import file_hash
        result=dict(actor_updated=False,world_model_updated=False,one_step_only=True,
                    source_sha256=file_hash(a.checkpoint),model_sha256=file_hash(a.contact_model),
                    intervention_delta_rad=a.delta,
                    terrain='stairs_inv',terrain_level=8,command=[1.,0.,0.],cases={},
                    prefix_abs_tolerances=PREFIX_ABS_TOLERANCES,
                    event_selection='First baseline step with >=min-active gait gates and at least one active FR shank-only contact',
                    interpretation='Event-conditioned one-step response with bounded numerical prefix drift and repeated baseline control')
        with tempfile.TemporaryDirectory(prefix='lite3_action_response_') as temp:
            for seed in a.seeds:
                seed_raw={}
                trigger_step=None
                for name,_,_ in cases:
                    raw_path=Path(temp)/f'{seed}_{name}.pt'
                    command=[sys.executable,'-u',str(Path(__file__).resolve()),
                             '--checkpoint',a.checkpoint,'--contact-model',a.contact_model,
                             '--output',a.output,'--seeds',str(seed),'--num-envs',str(a.num_envs),
                             '--max-seconds',str(a.max_seconds),'--min-active',str(a.min_active),
                             '--delta',str(a.delta),
                             '--device',a.device,'--worker','--case',name,
                             '--raw-output',str(raw_path)]
                    if trigger_step is not None:
                        command.extend(['--trigger-step',str(trigger_step)])
                    worker_env=child_environment()
                    worker_env['PYTHONHASHSEED']=str(seed)
                    subprocess.run(command,env=worker_env,check=True)
                    seed_raw[name]=torch.load(raw_path,map_location='cpu',weights_only=False)
                    if name=='baseline':
                        trigger_step=seed_raw[name]['step']
                matched,matching=verify_same_prefix(seed_raw)
                if matching['matched_active']<a.min_active:
                    raise RuntimeError(f'Seed {seed}: only {matching["matched_active"]} of '
                                       f'{matching["baseline_active"]} active environments have matched prefixes; '
                                       f'need {a.min_active}. Matching details: {json.dumps(matching)}')
                null_control=verify_null_response(seed_raw,matched)
                result['cases'][str(seed)]=summarize_counterfactual(seed_raw,matched,a.min_active)
                result.setdefault('prefix_matching',{})[str(seed)]=matching
                result.setdefault('null_control',{})[str(seed)]=null_control
                print(f'seed {seed}: bounded-prefix and repeated-baseline checks passed',flush=True)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
        print(f'Saved {out}. One-step response only; no actor optimization.',flush=True)
        return
    if a.case is None or a.raw_output is None or len(a.seeds)!=1:
        p.error('--worker requires one --case, one --seeds value, and --raw-output')
    if a.case!='baseline' and a.trigger_step<0:
        p.error('Non-baseline workers require --trigger-step')
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
                                                foot0,scale,-a.delta,0.)
                        active=(gate>.01)
                        if a.trigger_step>=0 and step==a.trigger_step and int(active.sum())<a.min_active:
                            raise RuntimeError(f'Seed {seed}, {name}: only {int(active.sum())} active envs at baseline trigger step {step}')
                        eligible=int(active.sum())>=a.min_active
                        evaluate=eligible and (a.trigger_step<0 or step==a.trigger_step)
                        if eligible:
                            pre_gpu=dict(state=s0,pose=p0,action=action,previous_action=prev,
                                         context=context,foot_contact=foot0,command=cmd,gate=gate)
                            pre_gpu={key:value.detach().clone() for key,value in pre_gpu.items()}
                        if evaluate:
                            adjusted,_=front_leg_probe(action,s0,p0,cmd,term.height_field,
                                                       foot0,scale,hipy,knee)
                        else:
                            adjusted=action
                        if eligible:
                            tested_action=adjusted.detach().clone()
                        obs,_,terminated,truncated,_=env.step(adjusted)
                        done=terminated|truncated
                        if eligible:
                            feet1,_=read_contacts(env.scene['feet_ground_contact'])
                            shanks1,_=read_shank_contacts(env.scene['shanks_ground_contact'])
                            shank_only=shanks1[:,0] & ~feet1[:,0]
                            selected=evaluate and (a.trigger_step>=0 or baseline_contact_event(active & ~done,shank_only,a.min_active))
                        else:
                            selected=False
                        if selected:
                            pre={key:value.cpu() for key,value in pre_gpu.items()}
                            relief=tr.relief(pre_gpu['pose'][:,:2],pre_gpu['pose'][:,2],term.height_field)
                            _,_,logits=wm.predict(pre_gpu['state'],tested_action,pre_gpu['previous_action'],
                                                  pre_gpu['context'],relief)
                            raw=dict(
                                step=step,pre=pre,
                                active=active.detach().cpu().clone(),
                                done=(terminated|truncated).detach().cpu().clone(),
                                prob=logits[:,8].sigmoid().detach().cpu().clone(),
                                actual=shank_only.detach().cpu().clone(),
                                foot=feet1[:,0].detach().cpu().clone(),
                                metadata=dict(step=step,active=int(active.sum()),
                                    falls=int(terminated.sum()),
                                    active_fr_shank_only=int((active & shank_only).sum()),
                                    action_target_hipy_delta_rad=hipy,action_target_knee_delta_rad=knee,
                                    mean_gate=float(gate[active].mean()),
                                    fr_hipy_action_delta=float((tested_action[active,4]-pre_gpu['action'][active,4]).mean()),
                                    fr_knee_action_delta=float((tested_action[active,5]-pre_gpu['action'][active,5]).mean())))
                            event=True
                            break
                        prev=torch.where(done[:,None],torch.zeros_like(adjusted),adjusted)
                        if done.any(): obs,_=env.reset(env_ids=done.nonzero().flatten())
                if event is None:
                    raise RuntimeError(f'No FR shank-only baseline event with {a.min_active} active envs in seed {seed}; extend --max-seconds or --num-envs')
                print(f'seed {seed}, {name}: intervention at step {step}; active={int(active.sum())}, shank-only={int((active & shank_only).sum())}',flush=True)
            finally:
                env.close()
        torch.save(raw,a.raw_output)


if __name__=='__main__': main()
