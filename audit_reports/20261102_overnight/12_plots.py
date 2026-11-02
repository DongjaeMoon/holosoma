# Standalone scientific figures from existing measurements only.
os.environ['MPLCONFIGDIR']=str(O/'matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':180})
plotdir=O/'plots';plotdir.mkdir(exist_ok=True)
palette=dict(biHAM='#146c94',sym_aug_HAM='#e38b20',HAM='#8a63b8',MLP='#267b4d')
def getrow(cohort,model,iteration,direction,speed):
 return next(r for r in metric_rows if (r['cohort'],r['model'],r['iteration'],r['direction'],r['speed'])==(cohort,model,iteration,direction,speed))
def savefig(fig,name):
 fig.savefig(plotdir/(name+'.png'),bbox_inches='tight');fig.savefig(plotdir/(name+'.svg'),bbox_inches='tight');plt.close(fig)
models=('biHAM','sym_aug_HAM','HAM','MLP')
labels={'biHAM':'biHAM','sym_aug_HAM':'HAM + symAug','HAM':'HAM','MLP':'MLP (v2)'}
fig,axes=plt.subplots(1,2,figsize=(12.6,4.3))
positions=np.arange(4);width=.34
for j,speed in enumerate((.3,.6)):
 values=[getrow('v2',m,10000,'right',speed)['rmse'] for m in models]
 bars=axes[0].bar(positions+(j-.5)*width,values,width,label=f'right {speed:.1f} m/s',color=['#3976a8','#da9b3d'][j])
 axes[0].bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
axes[0].set_xticks(positions,[labels[m] for m in models]);axes[0].set_ylabel('Planar velocity RMSE [m/s]');axes[0].set_ylim(0,.52);axes[0].legend();axes[0].set_title('Matched v2 conditions, checkpoint 10k')
for m in models:
 x=[-.6,-.3,.3,.6]
 y=[getrow('v2',m,10000,'right' if c<0 else 'left',abs(c))['mean_velocity'][1] for c in x]
 axes[1].plot(x,y,'o-',label=labels[m],color=palette[m])
axes[1].plot([-.65,.65],[-.65,.65],'k--',lw=1,label='perfect tracking');axes[1].axhline(0,color='#bbb',lw=.7);axes[1].axvline(0,color='#bbb',lw=.7)
axes[1].set(xlabel='Commanded lateral velocity [m/s]',ylabel='Measured mean lateral velocity [m/s]',title='Direction response and speed undershoot');axes[1].legend(fontsize=8)
fig.suptitle('Real differences exist; stable walking is not accurate velocity tracking',fontsize=12)
fig.tight_layout();savefig(fig,'01_v2_matched_comparison')
fig,axes=plt.subplots(1,2,figsize=(12.6,4.4))
groups=[('v2',m,10000,labels[m]) for m in models]+[('native','MLP',10000,'MLP native 10k'),('native','MLP',30000,'MLP native 30k')]
for ax,speed in zip(axes,(.3,.6)):
 vals=[getrow(c,m,k,'right',speed) for c,m,k,_ in groups];pos=np.arange(len(vals))
 bx=ax.bar(pos,[r['bias_mse'] for r in vals],color='#d28b39',label='Mean-velocity bias MSE')
 ax.bar(pos,[r['velocity_variability_mse'] for r in vals],bottom=[r['bias_mse'] for r in vals],color='#4d8ca9',label='Velocity variability MSE')
 ax.axvline(3.5,color='#555',linestyle='--',lw=1);ax.set_xticks(pos,[g[3] for g in groups],rotation=30,ha='right',fontsize=8)
 ax.set(title=f'Right command {speed:.1f} m/s',ylabel='Mean per-trial planar MSE [(m/s)^2]')
 for i,r in enumerate(vals):ax.text(i,r['mean_mse']+.003,f"vy={r['mean_velocity'][1]:.2f}",ha='center',fontsize=8)
axes[0].legend(fontsize=8);fig.suptitle('Error is often dominated by target-speed bias; native conditions differ',fontsize=12)
fig.tight_layout();savefig(fig,'02_error_decomposition')
fig,ax=plt.subplots(figsize=(10.3,4.5))
selected=[r for r in ar if r['cohort']=='v2']+[r for r in ar if r['cohort']=='native']
lab=[labels.get(r['model'],r['model']) if r['cohort']=='v2' else f"MLP native {r['iteration']//1000}k" for r in selected]
bars=ax.bar(np.arange(len(selected)),[r['full_policy_reflection_error']['rms'] for r in selected],color=[palette[r['model']] for r in selected])
ax.set_yscale('log');ax.set_ylim(5e-8,1);ax.set_xticks(np.arange(len(selected)),lab);ax.set_ylabel('RMS action reflection residual [policy action units]');ax.set_title('Saved weights: ||pi(Ro) - R pi(o)||, frozen normalizers')
for b,r in zip(bars,selected):ax.text(b.get_x()+b.get_width()/2,b.get_height()*1.4,f"{r['full_policy_reflection_error']['rms']:.2g}",ha='center',fontsize=9)
ax.text(.02,.93,'~96 sampled initial observations / checkpoint; no simulation',transform=ax.transAxes,fontsize=9)
fig.tight_layout();savefig(fig,'03_functional_symmetry')
fig,axes=plt.subplots(1,2,figsize=(12.5,4.1),sharey=True)
for m in models:
 steps=sorted({r['iteration'] for r in metric_rows if r['cohort']=='v2' and r['model']==m})
 values=[np.mean([getrow('v2',m,k,'right',s)['rmse'] for s in (.3,.6)]) for k in steps]
 axes[0].plot(np.array(steps)/1000,values,'o-',ms=3,color=palette[m],label=labels[m])
steps=sorted({r['iteration'] for r in metric_rows if r['cohort']=='native'})
for direction,color in [('left','#237e62'),('right','#be5e39')]:
 values=[np.mean([getrow('native','MLP',k,direction,s)['rmse'] for s in (.3,.6)]) for k in steps]
 axes[1].plot(np.array(steps)/1000,values,'o-',ms=2,color=color,label=direction)
for ax in axes:ax.set(xlabel='Training outer iterations [thousands]',ylabel='Mean planar RMSE, speeds 0.3 and 0.6 [m/s]');ax.legend(fontsize=8);ax.grid(alpha=.2)
axes[0].set_title('v2: same Gym / reflection normalization / Domi phase');axes[1].set_title('Native MLP: Sim / empirical normalization / native phase')
fig.suptitle('Learning curves: one training seed (43); rollouts do not replace training seeds',fontsize=12);fig.tight_layout();savefig(fig,'04_learning_curves')
fig,axes=plt.subplots(2,3,figsize=(12.5,6.5))
for col,(key,title) in enumerate((('vx','vx'),('vy','vy'),('yaw','yaw rate'))):
 for row,prefix in enumerate(('', 'normalized_')):
  data=columns[prefix+key];counts,edges=np.histogram(data,bins=60,density=True)
  axes[row,col].plot((edges[:-1]+edges[1:])/2,counts,color='#347a96',lw=1.5)
  axes[row,col].set(xlabel=('Raw '+title+' [m/s]' if key!='yaw' else 'Raw yaw rate [rad/s]') if row==0 else 'Normalized '+title,ylabel='Empirical histogram density')
  axes[row,col].axvline(0,color='#aaa',lw=.8)
axes[0,1].set_title('Raw vy >= 0 in all recorded rows')
axes[1,1].set_title('Negative normalized vy is not a right command')
fig.suptitle('Existing native MLP training CSV: environment 0 only, 30,001 time samples\n60-bin empirical densities, no KDE; zero-command atom appears as a bin spike',fontsize=11)
fig.tight_layout();savefig(fig,'05_command_distributions_env0')
print('FIGURES',len(list(plotdir.glob('*.png'))),flush=True)
