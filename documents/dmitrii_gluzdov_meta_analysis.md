## Cell 0 (markdown)

# Gemma 4: Public Agent Ideas and Evidence

## Abstract

Public repair agents vary in their architecture, prompts, search, editing, verification and use of time. A catalogue of 187 public notebooks groups 74 source mechanisms into nine families and relates their presence to recorded public scores. A paired local screen compares four configurations on 31 tasks from four repositories absent from earlier project evaluations. The budget-fit single agent and the budget-aware prompt each resolve nine tasks; adding edit-fallback and file-shortlist tools resolves eight with either prompt. These results leave small quality differences unresolved, while measured task times approach the competition's twelve-hour allowance. The released agent preserves flexonafft's public budget-aware bundle, combining its prompt with the established budget-fit configuration.

## The repair task

Each task supplies an issue report and a Python repository snapshot. The agent inspects the source, edits it and produces a patch that is checked by hidden tests. The organizers describe **about 120 tasks**, split between public and private scoring, with a **twelve-hour total limit** that includes sandbox setup and excludes patch validation. Sequential processing makes the average task time consequential: 120 tasks allow 360 seconds each before shared overhead. This is planning arithmetic, rather than an observed hidden task count. Every agent in a bundle shares one base model; this release uses the official quantized 31-billion-parameter Gemma 4 model. [Official evaluation](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/evaluation), [data and harness](https://www.kaggle.com/competitions/gemma-4-developer-agent/data).



## Cell 1 (code)

import hashlib, html, json, math, os, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display, HTML, Image, Markdown

INPUT = Path(os.environ.get('IDEA_MAP_INPUT', '/kaggle/input'))
OUT = Path(os.environ.get('IDEA_MAP_OUTPUT', '/kaggle/working'))
OUT.mkdir(parents=True, exist_ok=True)
def locate(filename):
    found = list(INPUT.rglob(filename))
    assert len(found) == 1, (filename, len(found))
    return found[0]
catalogue = pd.read_csv(locate('public_notebook_catalogue.csv'), keep_default_na=False)
ideas = pd.read_csv(locate('ideas.csv'), keep_default_na=False)
screen = pd.read_csv(locate('fresh_screen_task_metrics.csv'))
lb = pd.read_csv(locate('idea_map_leaderboard_observations.csv'))
registry = json.loads(locate('budget_aware_bundle.json').read_text())
protocol = json.loads(locate('fresh_screen_protocol.json').read_text())
assert len(catalogue) == 187 and catalogue.url.is_unique
assert len(screen) == 124 and not screen.duplicated(['task_id','configuration','seed']).any()
assert screen.groupby('configuration').task_id.nunique().eq(31).all()
assert screen.isna().sum().sum() == 0
order = ['Budget-fit single agent', 'Budget-aware prompt',
         'Budget-fit + edit-fallback and file-shortlist tools',
         'Budget-aware + edit-fallback and file-shortlist tools']
short = ['Budget-fit', 'Budget-aware', 'Budget-fit + tools', 'Budget-aware + tools']
family_order = ['Architecture','Prompt techniques','Localization','Editing','Verification','Budgets','Sampling','Model variant','Adapters']
tag_family = ideas.set_index('mechanism_tag').family.to_dict()
catalogue['families'] = catalogue.mechanism_tags.map(lambda s: {tag_family[t] for t in s.split(';') if t})
catalogue['score'] = pd.to_numeric(catalogue.best_score, errors='coerce')
summary = pd.DataFrame([{'Family':f,'Notebooks':sum(f in x for x in catalogue.families),
    'With a score':int(catalogue.loc[catalogue.families.map(lambda x:f in x),'score'].notna().sum()),
    'Best public score':catalogue.loc[catalogue.families.map(lambda x:f in x),'score'].max()} for f in family_order])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':11,'figure.facecolor':'#fbfcfe','axes.facecolor':'#fbfcfe','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'#fbfcfe'})
INK='#19334a'; TEAL='#167d8d'; ORANGE='#cf7e28'; GRAY='#607080'
def figure_done(fig, name):
    fig.savefig(OUT/name,dpi=165,bbox_inches='tight')
    plt.close(fig)
    display(Image(filename=str(OUT/name)))
def table(frame, formats=None):
    display(HTML(frame.to_html(index=False,escape=False,float_format=lambda x:f'{x:.2f}',formatters=formats or {})))



## Cell 2 (code)

fig, (a,b) = plt.subplots(1,2,figsize=(13,7.5),gridspec_kw={'width_ratios':[1,1.55]})
fig.subplots_adjust(left=.16,right=.94,top=.76,bottom=.20,wspace=.23)
fig.suptitle('Public ideas → paired evidence → one released agent',x=.5,y=.98,fontsize=18,color=INK,weight='bold')
fig.text(.5,.89,'187 notebooks  •  74 source mechanisms  •  4 configurations × 31 tasks',ha='center',fontsize=12,color=GRAY)
y=np.arange(len(family_order))
a.barh(y,summary.Notebooks,color=TEAL,height=.65)
a.set_yticks(y,family_order);a.set_ylim(8.5,-.5);a.set_xlim(0,190)
a.set_xlabel('Notebooks (count)');a.set_ylabel('Mechanism family')
a.set_title('Source presence',loc='left',pad=12)
for i,n in enumerate(summary.Notebooks):a.text(n+2,i,str(n),va='center',fontsize=10)
bin_names=['No score','0–0.09','0.10–0.14','0.15–0.19','0.20+']
def score_bin(x):
    if pd.isna(x):return 0
    return 1 if x<.1 else 2 if x<.15 else 3 if x<.2 else 4
matrix=np.array([[sum(f in r.families and score_bin(r.score)==j for r in catalogue.itertuples()) for j in range(5)] for f in family_order])
im=b.imshow(matrix,cmap='Blues',aspect='auto',vmin=0,vmax=matrix.max())
b.set_yticks(y,[]);b.set_xticks(range(5),bin_names,rotation=20,ha='right')
b.set_xlabel('Best public score per notebook (fraction)');b.set_title('Association with recorded score badges',loc='left',pad=12)
for i in range(9):
    for j in range(5):b.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()*.56 else INK,fontsize=11)
cbar=fig.colorbar(im,ax=b,fraction=.04,pad=.03);cbar.set_label('Notebooks (count)')
fig.text(.03,.055,'Figure 1. Families overlap. Tags include discussion, proposals and disabled mechanisms; scores do not isolate their effects.\nSource: public notebook catalogue and mechanism annotations, observed 14 October 2026.',fontsize=10,color=GRAY,linespacing=1.6)
figure_done(fig,'figure_1_public_idea_map.png')



## Cell 3 (markdown)

*Figure 1. The map summarizes source presence and recorded public score badges. Axes and colour show notebook counts and score fractions. Source: the [public catalogue and mechanism annotations](https://www.kaggle.com/datasets/dmitriigluzdov/gemma-4-serving-and-lora-verification-kit), observed 14 October 2026.*

## A map of public ideas

**Fact:** the snapshot contains 187 distinct notebook URLs, 49 observed score badges and 138 missing scores. It has no duplicate URLs. Seven licences remain unverified, and four output endpoints were unavailable; source inspection still covered all 187 notebooks. Missing scores stay missing. The catalogue covers the public listing captured on the observation date, not every future notebook or every historical revision.

The nine families describe different parts of a repair: architecture assigns roles; prompts organize decisions; localization finds relevant code; editing changes it; verification checks it; budgets limit work; sampling controls generation; model variants include model and serving investigations; adapters change weights. Counts below are distinct notebooks **mentioning a mechanism in that family**. They include source proposals, inherited declarations and explicit prohibitions. For example, graph tools appear in the budget-fit declaration even though its prompt prohibits their use; adapter text can discuss an option without shipping weights. The map therefore measures visible ideas, not tool execution.



## Cell 4 (code)

table(summary,{'Best public score':lambda x:f'{x:.2f}'})



## Cell 5 (markdown)

**Fact:** a public score belongs to one notebook revision and one run. A notebook's best badge may refer to an older revision than the inspected source. The maximum score in a family is consequently an association, not the contribution of that mechanism. Forks share ideas and are not independent replications. Broad categories often inherit the same maximum from one notebook.

The budget-fit family is a useful concrete example. verracodeguacas supplies a compact single-agent configuration. Matterhorn3838's prompt emphasizes exact issue requirements and nearby implementation patterns. flexonafft adds bounded batches of independent searches, small caller and test views, reuse of discovered paths, and recovery from failed edits. The released budget-aware bundle preserves the budget-fit configuration and changes only its system prompt. These are credited prompt antecedents, rather than experimentally isolated steps. [Budget-fit source](https://www.kaggle.com/code/verracodeguacas/gemma-4-agent-budget-fit-single-agent), [issue-focused prompt source](https://www.kaggle.com/code/matterhorn3838/gemma-4-superagent), [budget-aware source](https://www.kaggle.com/code/flexonafft/gemma-4-budget-aware-coding-agent-v2?scriptVersionId=357955943).



## Cell 6 (code)

fig,ax=plt.subplots(figsize=(12.5,6.6))
fig.subplots_adjust(left=.10,right=.97,top=.78,bottom=.31)
fig.suptitle('A prompt lineage on a fixed operating budget',y=.97,fontsize=18,weight='bold',color=INK)
fig.text(.5,.87,'8 minutes per task  •  28 counted calls  •  80 turns  •  4,096 thinking tokens',ha='center',fontsize=12,color=GRAY)
x=np.arange(3);scores=[.18,.17,.22]
ax.set_ylim(.10,.25);ax.set_xlim(-.5,2.5)
ax.set_ylabel('Recorded best public score (fraction)')
ax.set_xlabel('Credited prompt ancestry (categorical)',labelpad=12)
ax.set_xticks(x,['Budget-fit single agent\nverracodeguacas','Issue-focused prompt\nMatterhorn3838','Budget-aware prompt\nflexonafft'])
ax.grid(axis='y',alpha=.18)
for i in range(2):ax.annotate('',xy=(x[i+1]-.12,scores[i+1]),xytext=(x[i]+.12,scores[i]),arrowprops={'arrowstyle':'->','lw':1.8,'color':GRAY,'linestyle':'--'})
ax.scatter(x,scores,s=160,c=[GRAY,GRAY,TEAL],zorder=3)
for i,s in enumerate(scores):ax.text(i,s+.012,f'{s:.2f}',ha='center',fontsize=14,weight='bold',color=INK)
fig.text(.10,.145,'Prompt additions: exact requirements → sibling patterns → bounded navigation and edit recovery',fontsize=11,color=INK)
fig.text(.03,.045,'Figure 2. Arrows denote credited ancestry, not causal gains. The final bundle changes only the reference system prompt.\nSource: public sources by verracodeguacas, Matterhorn3838 and flexonafft; catalogue observed 14 October 2026.',fontsize=10,color=GRAY,linespacing=1.6)
figure_done(fig,'figure_2_budget_fit_lineage.png')



## Cell 7 (markdown)

*Figure 2. Credited ancestry and recorded public score fractions. Scores were observed on the source pages, not reproduced as a three-step experiment. Source: the three public notebooks linked above and the [bundle attribution registry](https://www.kaggle.com/datasets/dmitriigluzdov/gemma-4-serving-and-lora-verification-kit).*

## What changes the score: measured comparisons

### Public runs

The budget-fit reference and four modifications produced the following public observations. Each modification has one measured run. The unchanged coder–analyzer archive also has an author-reported score and two project runs.



## Cell 8 (code)

comparison = lb[lb.configuration!='Coder–analyzer reference'].sort_values('date')[['configuration','public_score','date']].rename(columns={'configuration':'Configuration','public_score':'Public score','date':'Date'})
table(comparison,{'Public score':lambda x:f'{x:.2f}'})
spread = lb[lb.configuration=='Coder–analyzer reference'][['date','public_score','evidence_type']].rename(columns={'date':'Date','public_score':'Same archive: public score','evidence_type':'Evidence'})
table(spread,{'Same archive: public score':lambda x:f'{x:.2f}'})



## Cell 9 (markdown)

**Fact:** the identical coder–analyzer archive spans **0.06–0.12** across these observations. The 0.12 value is the original author's reported result; 0.10 and 0.06 are project runs. Dates and scorer conditions differ, so this spread combines possible execution variability and scorer changes. It is not an estimate of sampling noise alone. The budget-fit prompt and tool modifications show no established gain in these sparse observations; score gaps of 0.01 cannot support a causal ranking here. [Dated observations](https://www.kaggle.com/datasets/dmitriigluzdov/gemma-4-task-soundness-and-local-results), [Roman Rozen's reference](https://www.kaggle.com/code/romanrozen/gemma-eda-baseline-for-a-start-lb-top-1).

### Paired local screen

**Fact:** all four configurations completed the same **31 tasks at seed 1**, giving 124 outcomes with no missing outcomes or duplicate task–configuration pairs. The repositories were pyairtable (7 tasks), django-axes (7), polyfactory (13) and bandit (4); none had appeared in the earlier project evaluations. Empty-patch controls resolved 0/31 and reference-patch controls resolved 31/31. This was a development screen used for comparison and selection. The second seed and separate model confirmation were not run.

All configurations retained the same model, sampling and eight-minute limit. A100 execution used one model replica and eight concurrent clients. Task times include agent work, setup/process overhead and teardown, while excluding grading and shared server startup. They are not measured sequential runtimes on the competition's four L4 GPUs. [Per-task measurements and frozen protocol](https://www.kaggle.com/datasets/dmitriigluzdov/gemma-4-task-soundness-and-local-results).



## Cell 10 (code)

stats=screen.groupby('configuration').agg(tasks=('task_id','size'),resolved=('resolved','sum'),mean_seconds=('agent_seconds','mean'),p95_seconds=('agent_seconds',lambda x:x.quantile(.95)),timeouts=('agent_timeout','sum'),other_errors=('other_agent_error','sum')).reindex(order)
stats['projection_hours']=stats.mean_seconds*120/3600
local_table=pd.DataFrame({'Configuration':order,'Resolved / 31':[f'{n}/31' for n in stats.resolved],
    'Mean (s)':stats.mean_seconds.values,'95th percentile (s)':stats.p95_seconds.values,
    'Agent timeouts':stats.timeouts.values,'Other agent errors':stats.other_errors.values,
    '120 × mean (h)':stats.projection_hours.values})
table(local_table,{'Mean (s)':lambda x:f'{x:.1f}','95th percentile (s)':lambda x:f'{x:.1f}','120 × mean (h)':lambda x:f'{x:.2f}'})
pivot=screen.pivot(index='task_id',columns='configuration',values='resolved')
pairs=[]
for candidate in order[1:]:
    wins=int(((pivot[candidate]==1)&(pivot[order[0]]==0)).sum())
    losses=int(((pivot[candidate]==0)&(pivot[order[0]]==1)).sum())
    n=wins+losses
    p=min(1,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/(2**n)) if n else 1
    pairs.append({'Candidate versus budget-fit':candidate,'Candidate-only wins':wins,'Budget-fit-only wins':losses,'Net wins':wins-losses,'Exact sign-test p':p})
table(pd.DataFrame(pairs),{'Exact sign-test p':lambda x:f'{x:.3f}'})



## Cell 11 (code)

fig,(a,b)=plt.subplots(1,2,figsize=(13,6.9),gridspec_kw={'width_ratios':[1,1.35]})
fig.subplots_adjust(left=.18,right=.97,top=.78,bottom=.23,wspace=.15)
fig.suptitle('Similar repair counts; little room in the time budget',y=.97,fontsize=18,color=INK,weight='bold')
fig.text(.5,.88,'Same 31 fresh-repository tasks  •  seed 1  •  local development screen',ha='center',fontsize=12,color=GRAY)
y=np.arange(4)
a.barh(y,stats.resolved,color=[GRAY,TEAL,GRAY,TEAL],height=.6)
a.set_yticks(y,short);a.set_ylim(3.5,-.5);a.set_xlim(0,12)
a.set_xlabel('Resolved tasks (count out of 31)');a.set_ylabel('Configuration');a.set_xticks([0,3,6,9,12])
for i,n in enumerate(stats.resolved):a.text(n+.15,i,f'{n}/31',va='center',fontsize=11)
b.scatter(stats.mean_seconds,y,s=80,color=TEAL,label='Mean',zorder=3)
b.scatter(stats.p95_seconds,y,s=65,color=ORANGE,marker='D',label='95th percentile',zorder=3)
for i,row in enumerate(stats.itertuples()):
    b.plot([row.mean_seconds,row.p95_seconds],[i,i],color='#c7d3db',lw=2)
    left=row.mean_seconds<360
    b.text(row.mean_seconds+(-5 if left else 5),i-.17,f'{row.mean_seconds:.0f}',ha='right' if left else 'left',fontsize=10,color=INK)
b.axvline(360,color='#ae4141',ls='--',lw=1.5,label='360 s planning allowance')
b.set_yticks(y,[]);b.set_xlim(300,525);b.set_ylim(3.5,-.5)
b.set_xlabel('Agent + setup/teardown time (seconds per task)')
b.legend(loc='lower right',bbox_to_anchor=(1,1.02),fontsize=9,frameon=False,ncol=1)
b.grid(axis='x',alpha=.15)
fig.text(.03,.06,'Figure 3. The 360-second line is 12 hours / 120 tasks before shared overhead. Local A100 timings are not L4 predictions.\nSource: 124 frozen local outcomes across four repositories, completed 15 October 2026; grading and shared startup excluded.',fontsize=10,color=GRAY,linespacing=1.6)
figure_done(fig,'figure_3_quality_and_time.png')



## Cell 12 (markdown)

*Figure 3. Resolution counts and measured time per task. The dashed line is a planning allowance, not a measured scorer threshold per task. Source: [the complete local screen](https://www.kaggle.com/datasets/dmitriigluzdov/gemma-4-task-soundness-and-local-results).*

**Fact:** the budget-aware prompt has two candidate-only wins and two reference-only wins. The tool configurations have one win versus two losses, and two wins versus three losses. Each exact two-sided sign test gives p = 1.0. These task-level tests ignore dependence within four repositories and multiple comparisons; p = 1 does not establish equivalence. Repository weighting can also change the descriptive ordering.

The arithmetic projection for 120 tasks is 11.63 hours for budget-fit, 11.99 for budget-aware, 12.46 for budget-fit with tools and 13.02 for budget-aware with tools. These values exclude shared startup and do not transfer directly to competition hardware. **Hypothesis:** time per task is the binding practical constraint for this family. A scorer-like sequential timing study would test that hypothesis. The current quality measurements do not resolve a prompt or tool advantage; their differences remain compatible with noise.

### Were the added tools used?



## Cell 13 (code)

usage=screen.groupby('configuration')[['shortlist_loaded','shortlist_loads','shortlist_calls','shortlist_successes','edit_fallback_loads','edit_fallback_calls','edit_fallback_commands']].sum().reindex(order)
usage_table=pd.DataFrame({'Configuration':order,'Shortlist tasks / 31':[f'{n}/31' for n in usage.shortlist_loaded],
    'Shortlist loads':usage.shortlist_loads.values,'Script calls':usage.shortlist_calls.values,'Successful calls':usage.shortlist_successes.values,
    'Edit fallback: loads / calls / commands':[f'{r.edit_fallback_loads} / {r.edit_fallback_calls} / {r.edit_fallback_commands}' for r in usage.itertuples()]})
table(usage_table)



## Cell 14 (markdown)

**Fact:** the file-shortlist tool was loaded on eight tasks with the budget-fit prompt. All eight first passed an incorrect repository root and then retried successfully, producing eight path failures and eight successful calls. The budget-aware configuration with added tools declared both but used neither. Edit fallback was never exercised. **Hypothesis:** the local path mismatch consumed useful budget; the screen cannot isolate its effect or establish that it would occur in the competition container. These outcomes therefore do not show that the unused edit-fallback mechanism is ineffective.

Agent session timeouts numbered 6, 6, 9 and 13 across the four configurations. One additional budget-aware run exhausted the call limit. Timeout recovery can still yield a patch that passes grading, so timeout and resolution are not mutually exclusive. Grading errors and grading timeouts were absent.

## The submitted agent

The packaged agent combines the budget-fit operating profile with flexonafft's budget-aware prompt. It identifies literal requirements, locates the relevant implementation and nearby callers or tests, edits a small region, performs targeted verification and finishes with a patch. Independent searches share a bounded command; dependent searches wait for evidence. Failed edits trigger a fresh source view and then a guarded unique-match replacement. The declared graph tools are discouraged by the prompt. No added skills, subagents or learned adapters are included.

The release retains **eight minutes, 28 counted calls and 80 turns per task**, a 240-second command limit, temperature 0.2, top-p 0.95, an 8,192-token output cap and a 4,096-token thinking budget. Three source members are unchanged from the credited public bundle; the agent file declares the **12-billion-parameter quantized Gemma 4 model** instead of the 31-billion-parameter one. Relative to the budget-fit reference, the system prompt and the model differ. The archive is built deterministically on CPU without model inference or runtime internet access.

**Sampling observations:** the same bundle scored **0.15 at temperature 0, 0.18 at 0.2 and 0.12 at 0.4** in single public runs, and one identical third-party bundle scored 0.13 and 0.20 in two runs. Temperature is therefore not a lever that separates from leaderboard noise; 0.2 is retained.

**Model variation in this release:** all prompt and sampling changes on the 31B model stayed inside the noise, while time per task is the binding constraint for about 120 sequential tasks in twelve hours. This release changes the one quantity that sets time per token: it declares the permitted **12B** quantized model with the same prompt, tools and limits. A smaller model decodes faster and leaves more of each task's eight minutes for search, editing and checks, at the cost of weaker reasoning per step. The organizers list the 12B model as permitted and report a loading fix for it; this release is its first public measurement in this notebook, and the score is one draw from a noisy leaderboard.

**Release rationale:** the budget-aware source has the strongest observed public score in this catalogue, **0.22**, with the same configured time budget as the 0.18 budget-fit reference and the same local resolution count. This supports a practical choice under uncertainty, not a demonstrated improvement. Its local mean time was 10.6 seconds longer. The **score from this notebook is shown on the notebook page** when scoring is available; 0.22 remains the original author's result. [Budget-aware source and Apache 2.0 licence](https://www.kaggle.com/code/flexonafft/gemma-4-budget-aware-coding-agent-v2?scriptVersionId=357955943).



## Cell 15 (code)

members={}
for name,info in registry['members'].items():
    payload=locate(info['dataset_file']).read_bytes()
    assert hashlib.sha256(payload).hexdigest()==info['sha256']
    members[name]=payload
assert sorted(members)==['agent.yaml','configs/sampling.yaml','eval_config.yaml','prompts/system.md']
assert [n for n,b in members.items() if hashlib.sha256(b).hexdigest()!=registry['reference_member_sha256'][n]]==['prompts/system.md']
# Model variation tested in this release: the 12B quantized model instead of 31B (all other bytes unchanged).
source_agent=members['agent.yaml']
assert source_agent.count(b'model: gemma-4-31b-it-qat-w4a16-ct')==1
members['agent.yaml']=source_agent.replace(b'model: gemma-4-31b-it-qat-w4a16-ct',b'model: gemma-4-12b-it-qat-w4a16-ct')
archive=OUT/'submission.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
    for name,payload in sorted(members.items()):
        info=zipfile.ZipInfo(name,date_time=(2020,1,1,0,0,0))
        info.create_system=3;info.external_attr=0o100644<<16;info.compress_type=zipfile.ZIP_STORED
        z.writestr(info,payload)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert all(z.read(n)==b for n,b in members.items())
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
receipt={'sha256':digest,'members':{n:hashlib.sha256(b).hexdigest() for n,b in members.items()},'changed_from_budget_fit':['prompts/system.md','agent.yaml'],'changed_from_budget_aware_source':['agent.yaml: model gemma-4-31b-it-qat-w4a16-ct -> gemma-4-12b-it-qat-w4a16-ct'],'cpu_only':True,'model_inference':False,'source_archive_sha256':registry['source_archive_sha256']}
(OUT/'archive_validation.json').write_text(json.dumps(receipt,indent=2))
(OUT/'submission.sha256').write_text(digest+'  submission.zip\n')
display(HTML('<p><strong>Archive verified.</strong> Budget-aware public bundle with one model change: the 12B quantized model instead of 31B.</p>'))



## Cell 16 (markdown)

## Practical guidance and limitations

1. **Measure completion time before expanding the agent.** An eight-minute cap does not imply twelve-hour completion across about 120 tasks. Shared startup, sandbox setup and long-tail tasks need a measured allowance.
2. **Test one mechanism on fixed tasks and seeds.** Retain errors and timeouts, inspect paired wins and losses, and reserve new repositories for confirmation. The present screen is too small to settle modest effects.
3. **Check actual tool use and successful execution.** A declared tool cannot help if it is never called; a path error can consume the same budget as useful work. Trigger frequency, successful calls and repair quality answer different questions.
4. **Separate provenance from effectiveness.** Preserve the exact bundle and its licence. Public badges, source tags and author reports are useful evidence with different limits; none establishes private performance.

The unchanged public prompt has limitations: its final paragraph suggests checking after patch submission, although the harness terminates the agent loop on submission. Its claims about protected tests and graph usefulness are prompt policies, not complete descriptions of the harness. Keeping the source unchanged preserves the observed reference; it does not endorse every instruction as optimal. Verification must precede the final patch action. [Official harness reference](https://www.kaggle.com/competitions/gemma-4-developer-agent/data).

The catalogue is a dated, heuristic source map. Best badges can refer to older source, unscored notebooks can contain useful ideas, and inherited boilerplate inflates some families. The screen has one seed and four repository clusters, with no independent model confirmation. Public scores span dates and scorer conditions. No hidden-performance estimate or unmeasured probability is supplied.

## Credits and references

The released configuration and prompt are **Apache 2.0**: **verracodeguacas** for the [budget-fit core](https://www.kaggle.com/code/verracodeguacas/gemma-4-agent-budget-fit-single-agent), **Matterhorn3838** for the [issue-focused prompt antecedent](https://www.kaggle.com/code/matterhorn3838/gemma-4-superagent), and **Igor Zharov / flexonafft** for the [budget-aware prompt](https://www.kaggle.com/code/flexonafft/gemma-4-budget-aware-coding-agent-v2?scriptVersionId=357955943). The attribution chain also credits the [official starter by the competition organizers](https://www.kaggle.com/competitions/gemma-4-developer-agent/data). **Roman Rozen** supplies the [coder–analyzer reference](https://www.kaggle.com/code/romanrozen/gemma-eda-baseline-for-a-start-lb-top-1), also Apache 2.0. The [Apache 2.0 licence](https://www.apache.org/licenses/LICENSE-2.0) and attribution notices accompany the public bundle.

**Dmitry Gluzdov** supplies the local measurements, edit-fallback and file-shortlist tool experiments, and catalogue annotations. Original measurements and documentation are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); original code is Apache 2.0. Every mapped notebook is credited below by author, source link and observed licence. An unknown licence records missing evidence and grants no code reuse permission. No code from those sources is redistributed.

Data: [Task Soundness and Local Results](https://www.kaggle.com/datasets/dmitriigluzdov/gemma-4-task-soundness-and-local-results) and [Serving and LoRA Verification Kit](https://www.kaggle.com/datasets/dmitriigluzdov/gemma-4-serving-and-lora-verification-kit). Competition: [overview and evaluation](https://www.kaggle.com/competitions/gemma-4-developer-agent), [data and harness](https://www.kaggle.com/competitions/gemma-4-developer-agent/data). Related project study: [Task Soundness, Serving and Evaluation](https://www.kaggle.com/code/dmitriigluzdov/gemma-4-task-soundness-serving-and-evaluation).



## Cell 17 (code)

rows=[]
for r in catalogue.sort_values(['author','url']).itertuples():
    author=html.escape(r.author);url=html.escape(r.url,quote=True)
    families=', '.join(f for f in family_order if f in r.families) or 'No mapped mechanism'
    rows.append(f'<tr><td><a href="{url}">{author}</a></td><td>{html.escape(families)}</td><td>{html.escape(r.licence)}</td></tr>')
display(HTML('<details><summary>All 187 notebook sources: authors, mechanisms and licences</summary><table><thead><tr><th>Author and source link</th><th>Source families</th><th>Observed licence</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></details>'))
summary.to_csv(OUT/'mechanism_family_summary.csv',index=False)
local_table.to_csv(OUT/'local_screen_summary.csv',index=False)
pd.DataFrame(pairs).to_csv(OUT/'paired_comparisons.csv',index=False)


