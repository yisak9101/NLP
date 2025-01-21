import os
os.environ['OPENAI_API_KEY'] = 'YOUR SECRET KEY'
os.environ['GEMINI_API_KEY'] = 'YOUR SECRET KEY'

# CotQA_context
# import sys
# sys.path.append('')
# root = './root/'
#
# import joblib
# import numpy as np
# from agents import CoTAgent, ReflexionStrategy
# from util import summarize_trial, log_trial, save_agents
#
# hotpot = joblib.load('./data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)
#
# hotpot['supporting_paragraphs'] = None
# for ind, row in hotpot.iterrows():
#     supporting_articles = row['supporting_facts']['title']
#     articles = row['context']['title']
#     sentences = row['context']['sentences']
#     supporting_paragraphs = []
#     for article in supporting_articles:
#         supporting_paragraph = ''.join(sentences[np.where(articles == article)][0])
#         supporting_paragraphs.append(supporting_paragraph)
#     supporting_paragraphs = '\n\n'.join(supporting_paragraphs)
#     hotpot.at[ind, 'supporting_paragraphs'] = supporting_paragraphs
#
# print(ReflexionStrategy.__doc__)
# strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION
#
# from prompts import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt
# from fewshots import COT, COT_REFLECT
# agents = [CoTAgent(row['question'],
#                    row['supporting_paragraphs'],
#                    row['answer'],
#                    agent_prompt=cot_agent_prompt if strategy == ReflexionStrategy.NONE else cot_reflect_agent_prompt,
#                    cot_examples=COT,
#                    reflect_prompt=cot_reflect_prompt,
#                    reflect_examples=COT_REFLECT,
#                     ) for _, row in hotpot.iterrows()]
#
# n = 1
# trial = 0
# log = ''
#
# for i in range(n):
#     for agent in [a for a in agents if not a.is_correct()]:
#         agent.run(reflexion_strategy = strategy)
#         print(f'Answer: {agent.key}')
#     trial += 1
#     log += log_trial(agents, trial)
#     correct, incorrect = summarize_trial(agents)
#     print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')
#
# with open(os.path.join(root, 'CoT', 'context', strategy.value, f'{len(agents)}_questions_{trial}_trials.txt'), 'w') as f:
#     f.write(log)
# save_agents(agents, os.path.join(root, 'CoT', 'context', strategy.value, 'agents'))

# ReactQA
import sys, os
sys.path.append('..')
root  = './root/'

import joblib
from util import summarize_react_trial, log_react_trial, save_agents
from agents import ReactReflectAgent, ReactAgent, ReflexionStrategy

hotpot = joblib.load('./data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)

print(ReflexionStrategy.__doc__)

strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION
agent_cls = ReactReflectAgent if strategy != ReflexionStrategy.NONE else ReactAgent
agents = [agent_cls(row['question'], row['answer']) for _, row in hotpot.head(30).iterrows()]

n = 5
trial = 0
log = ''

for i in range(n):
    for agent in [a for a in agents if not a.is_correct()]:
        if strategy != ReflexionStrategy.NONE:
            agent.run(reflect_strategy = strategy)
        else:
            agent.run()
        print(f'Answer: {agent.key}')
    trial += 1
    log += log_react_trial(agents, trial)
    correct, incorrect, halted = summarize_react_trial(agents)
    print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}')

with open(os.path.join(root, 'ReAct', strategy.value, f'{len(agents)}_questions_{trial}_trials.txt'), 'w') as f:
    f.write(log)
save_agents(agents, os.path.join('ReAct', strategy.value, 'agents'))