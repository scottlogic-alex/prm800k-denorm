# prm800k-denorm

This repository is home to a script for converting OpenAI's [PRM800K](https://github.com/openai/prm800k/tree/main) process supervision dataset to a denormalized format in `.parquet` file type, for easier consumption.

Datasets converted with this tool have been published at [huggingface.co/Birchlabs](https://huggingface.co/Birchlabs).

## Why would I want process supervision?

See OpenAI's [blog post](https://openai.com/research/improving-mathematical-reasoning-with-process-supervision) and "Let's Verify Step by Step" [paper](https://arxiv.org/abs/2305.20050), but essentially it helps to improve a language model's reasoning and human alignment.

If you only reward a language model for correct answers, then incentivizes "reward hacking" — reaching correct conclusions via incorrect reasoning. This results in a poor underlying understanding, which fails to generalize, and gives rise to logical mistakes (often called "hallucinations").

So, rather than the usual "outcome supervision": let's try "process supervision": reward the language model for making good _progress_ towards solutions. This will improve its reasoning.  
We encourage this, by teaching it to complete maths problems step-by-step, showing its workings.

Our goal with this repository is to give you access to a step-by-step dataset, so that you can train your language model in this way.

The dataset is semantic, so that you can template it into your prompt style however works best for you.

## Dataset types

In total, the datasets are:

- Solutions only
  - phase 1
    - [train](https://huggingface.co/datasets/Birchlabs/openai-prm800k-phase1_train-solutions-only)
    - [test](https://huggingface.co/datasets/Birchlabs/openai-prm800k-phase1_test-solutions-only)
  - phase 2
    - [train](https://huggingface.co/datasets/Birchlabs/openai-prm800k-phase2_train-solutions-only)
    - [test](https://huggingface.co/datasets/Birchlabs/openai-prm800k-phase2_test-solutions-only)
- Stepwise best
  - phase 1
    - [train](https://huggingface.co/datasets/Birchlabs/openai-prm800k-phase1_train-stepwise-best)
    - [test](https://huggingface.co/datasets/Birchlabs/openai-prm800k-phase1_test-stepwise-best)
  - phase 2
    - [train](https://huggingface.co/datasets/Birchlabs/openai-prm800k-phase2_train-stepwise-best)
    - [test](https://huggingface.co/datasets/Birchlabs/openai-prm800k-phase2_test-stepwise-best)
- Stepwise critique
  - phase 1
    - [train](https://huggingface.co/datasets/Birchlabs/openai-prm800k-phase1_train-stepwise-critique)
    - [test](https://huggingface.co/datasets/Birchlabs/openai-prm800k-phase1_test-stepwise-critique)
  - phase 2
    - [train](https://huggingface.co/datasets/Birchlabs/openai-prm800k-phase2_train-stepwise-critique)
    - [test](https://huggingface.co/datasets/Birchlabs/openai-prm800k-phase2_test-stepwise-critique)

### Solutions only

```
# the problem statement (e.g. "what is x, given 2x = 1?")
instruction: str

# history of responses that the bot emitted before reaching this final step
responses: List[str]

# response emitted for the final step in the conversation.
# it is accompanied by an "answer" subsection, which we capture separately
next_response: str

# answer subsection of the final response
answer: str
```

Only conversations which achieved `"finish_reason": "solution"` are retained.

We provide only one record per conversation — the final step.  
This record contains a list of all the steps which led to that final step. So you have the whole conversation available.

You could use this to create the following kind of training data:

- **Source**: initial instruction
- **Target**: all steps of the conversation, including the final step and answer

#### Usage

Example of how to take this sample and turn it into a source->target training pair:

sample:  
```
instruction: "What is x, given 2x + 1 = 2?"

responses: [
    "Okay, let's first rearrange the equation to isolate the x.",
    "2x = 2 - 1, which simplifes to 2x = 1",
    "Now that the x term is isolated, let's divide both sides to eliminate its coefficient."
]

next_response: "x = 1/2"

answer: "0.5"
```

Example Alpaca-style source prompt:

```
Below is an instruction that describes a task. Write responses which progress toward a solution to the request. Indicate your final answer under a heading Final Answer.

### Instruction:
What is x, given 2x + 1 = 2?

### Response:
Okay, let's first rearrange the equation to isolate the x.

### Response:
2x = 2 - 1, which simplifes to 2x = 1

### Response:
Now that the x term is isolated, let's divide both sides to eliminate its coefficient.

### Response:
```

Example Alpaca-style target prompt:

```
x = 1/2

# Answer
0.5
```

### Stepwise best

```
# the problem statement (e.g. "what is x, given 2x = 1?")
instruction: str

# any responses that were emitted in prior conversation steps
# empty list means it's the first conversation turn
responses: List[str]

# the response emitted for the current step in the conversation
# if this is the final response in the conversation: the response may be accompanied by
# an answer subsection. we separate this out into the `answer` field below.
next_response: str

# usually None, but if this is filled in: the final response of the conversation has occurred,
# and has a subsection declaring its overall answer. this field captures that overall answer.
answer: Optional[str]

# often, there are conversation steps in which the human evaluator had to intervene,
# as no generated response was satisfactory.
is_human_response: bool
```

We provide a record per productive conversation turn.  
Whichever is the preferred completion for that conversation turn (i.e. the `chosen_completion` if a bot response was preferred, or a `human_completion` otherwise), is the one we use.

You could use this to create the following kind of training data:

**Source**: initial instruction + 0-to-many steps of conversation so far
**Target**: next step of conversation (doesn't necessarily get you to a complete solution)

#### Usage

Example of how to take this sample and turn it into a source->target training pair:

sample:  
```
instruction: "What is x, given 2x + 1 = 2?"

responses: [
    "Okay, let's first rearrange the equation to isolate the x.",
    "2x = 2 - 1, which simplifes to 2x = 1",
]

next_response: "Now that the x term is isolated, let's divide both sides to eliminate its coefficient."

answer: None

is_human_response: False
```

Example Alpaca-style source prompt:

```
Below is an instruction that describes a task. Write responses which progress toward a solution to the request. Indicate your final answer under a heading Final Answer.

### Instruction:
What is x, given 2x + 1 = 2?

### Response:
Okay, let's first rearrange the equation to isolate the x.

### Response:
2x = 2 - 1, which simplifes to 2x = 1

### Response:
```

Example Alpaca-style target prompt:

```
Now that the x term is isolated, let's divide both sides to eliminate its coefficient.
```

Notice how this time there was no "answer" subsection, because this conversation step does not propose a solution. It can happen though, so check whether answer is `None`.

### Stepwise critique

```
# the problem statement (e.g. "what is x, given 2x = 1?")
instruction: str

# any responses that were emitted in prior conversation steps
# empty list means it's the first conversation turn
responses: List[str]

# the response emitted for the current step in the conversation
# if this is the final response in the conversation: the response may be accompanied by
# an answer subsection. we separate this out into the `answer` field below.
next_response: str

# usually None, but if this is filled in: the final response of the conversation has occurred,
# and has a subsection declaring its overall answer. this field captures that overall answer.
answer: Optional[str]

# often, there are conversation steps in which the human evaluator had to intervene,
# as no generated response was satisfactory.
is_human_response: bool

# whether an answer was provided **and** was correct
is_solution: bool

# tells us if this completion was rated as the best option by the human evaluator
is_preferred_response: bool

#   -1 = counterproductive
#    0 = unproductive
#    1 = productive
# None = human answer (you can treat this as 1)
rating: Optional[int]
```

We provide a record per proposed completion, per conversation turn.  
This includes human ratings.

You could use this to create the following kind of training data:

**Source**: initial instruction + 0-to-many steps of conversation so far
**Target**: a possible completion and rating

The data here includes both good and bad rated samples.

You could just keep the good ones. That would be similar to what Stepwise best gives you, but instead of "best" it gives you a wider "any answer which got a good rating, even if it wasn't the best one". So, more data (lower quality).

But an even better usage is to use this to train a critic. @crowsonkb has demonstated that if you have a classifier, you can [guide your sampling](https://gist.github.com/crowsonkb/af6135392cc1627f40b03456aa90810c) of the language model, generating a few candidate next-tokens and picking the one that your classifier prefers.  
So you could use this to train a critic, to guide another language model to employ more reasoning in its responses.

### Usage

You'd format this the same as stepwise best.

## Setup

_Note: if you're happy with the [published datasets](https://huggingface.co/Birchlabs), then there's no need for you to get this repository set up yourself — these Setup instructions are for tinkerers who wish to try the export themselves, or change the format_.

### Get OpenAI data

Copy the PRM800K and MATH datasets into the `prm800k` directory, in accordance with [`prm800k/README.md`](prm800k/README.md).

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Run the converter

```bash
python -m scripts.convert
```

This should output some .parquet files to a directory `out`, at the root of the repository.