"""Export the relaxed, ready and normal spring/tendon operating points."""
import argparse
import json
from pathlib import Path
import numpy as np
from .config import ModelConfig
from .models.spring import ReturnSpring, static_operating_point
from .models.tendon import Tendon


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/spring_stages'))
    parser.add_argument('--normal-angle-deg',type=float,default=40.)
    args=parser.parse_args()
    config=ModelConfig.load(args.config)
    spring=ReturnSpring(config);tendon=Tendon.from_config(config)
    if config.spring.mode != 'extension':
        parser.error('This demonstration requires spring.mode=extension')
    names=('Initial','Ready','Normal actuation')
    angles=(spring.relaxed_angle,config.joint.equilibrium_rad,np.deg2rad(args.normal_angle_deg))
    rows=[dict(stage=name,**static_operating_point(config,angle)) for name,angle in zip(names,angles)]
    args.output.mkdir(parents=True,exist_ok=True)
    config.save(args.output/'config.json')
    (args.output/'stages.json').write_text(json.dumps(rows,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,6),sharex=True,sharey=True)
    for ax,row,q in zip(axes,rows,angles):
        origin=np.asarray(config.joint.origin_xy_m)
        tip=origin+config.joint.distal_length_m*np.array([np.cos(q-config.joint.equilibrium_rad),np.sin(q-config.joint.equilibrium_rad)])
        finger=np.array([[0,0],origin,tip])[:,::-1]
        ax.plot(finger[:,0],finger[:,1],'-o',lw=6,color='gray')
        route=np.array([tendon.spool_position,*tendon.guide_positions(q)])[:,::-1]
        ax.plot(route[:,0],route[:,1],'o--',color='royalblue',label='Tendon / P1-P4')
        anchors=np.array([spring.base,spring.moving_anchor(q)])[:,::-1]
        ax.plot(anchors[:,0],anchors[:,1],'o-',lw=3,color='orange',label='Extension spring')
        ax.add_patch(plt.Circle(tendon.spool_position[::-1],config.motor.spool_radius_m,fill=False))
        ax.set(title=f"{row['stage']} ({np.rad2deg(q):.0f}°)\nT={row['tension_N']:.2f} N; spring={row['spring_force_N']:.2f} N",
               xlabel='Lateral coordinate [m]',ylabel='Height [m]',aspect='equal',xlim=(-.065,.095),ylim=(-.025,.28))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Spring–tendon actuation: demonstration geometry, no external contact')
    fig.tight_layout();fig.savefig(args.output/'stages.png',dpi=160);plt.close(fig)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
