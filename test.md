RLHF is a technique we can use to try and better align an LLM's
output with user intention and preference.

In this first lesson, we're going to dive into a

conceptual overview of RLHF.

Let's get started.

Let's say that we want to tune a model on a summarization task.

We might start by gathering some text samples to

summarize and then have humans produce a summary

for each input.

So for example, here we have the input text,

before I go to university, I want to take a road trip in Europe.

I've lived in several European cities, but

there's still a lot I haven't seen, etc.

And then, we have a corresponding summary of that text.

The user wants to take a road trip in Europe before university.

They want to see as much as possible in a short time,

and they're wondering if they should go to places that

are significant from their childhood or places they

have never seen. We can use these human-generated summaries

to create pairs of input text and summary, and

we could train a model directly on a bunch

of these pairs.

But the thing is, there's no one correct way to

summarize a piece of text.

Natural language is flexible, and there are often many

ways to say the same thing. For example, here's

an equally valid summary.

And in fact, there are many more valid summaries we could write.

Each summary might be technically correct,

but different people, different groups of people,

different audiences will all have a preference.

And preferences are hard to quantify.

Some problems like entity extraction or

classification have correct answers, but sometimes the task we

wanna teach the model doesn't have a clear objective best answer.

So, instead of trying to find the best summary for a particular piece

of input text, we're gonna frame this problem

a little differently.

We're going to gather information on human preferences, and

to do that, we'll provide a human labeler with two candidate
1:52
summaries and ask the labeler to pick which one they prefer.
1:56
And instead of the standard supervised
1:58
tuning process where we tune the model to map an input to
2:02
a single correct answer, we'll use reinforcement learning to
2:04
tune the model to map an input to a single correct answer, we'll use
2:04
reinforcement learning to tune the model to produce responses
2:08
that are aligned with human preferences.
2:10
So how does all this work?
2:11
Well, it's an evolving area of research and there are a lot of variations
2:15
and how we might implement RLHF specifically, but
2:18
the high level themes are the same.
2:21
RLHF consists of three stages.
2:24
First, we create a preference data set.
2:26
Then, we use this preference data set to train a
2:29
reward model with supervised learning. And then,
2:32
we use the reward model in a reinforcement learning
2:34
loop to fine tune our base large language model. Let's
2:37
look at each of these steps in detail.
2:39
And don't worry if you're totally new to reinforcement
2:42
learning.
2:42
You don't need any background for this course.
2:45
First things first, we're going to start with
2:47
the large language model that we want to tune. In
2:49
other words, the base LLM.
2:52
In this course, we're going to be tuning the
2:54
open source LLMA2 Model, and you'll get to see how that works
2:57
in a later lesson. But before we actually do any
3:00
model tuning, we're going to use this base LLM to
3:02
generate completions for a set of prompts.
3:06
So for example, we might send the input prompt,
3:08
summarize the following text, I want to start gardening,
3:11
but et cetera. And we would get the model to generate multiple
3:14
output completions for the same prompt.
3:17
And then, we have human labelers rate these completions.
3:20
Now, the first way you might think to do this is to have
3:23
the human labelers indicate on some absolute scale
3:26
how good the completion is. But this doesn't yield the best results
3:29
in practice because scales like this are subjective
3:32
and they tend to vary across people.
3:34
Instead, one way of doing this that's worked pretty well is to have the human
3:38
labeler compare two different output completions for the
3:42
same input prompt, and then specify which one they prefer.
3:47
This is the dataset that we talked about earlier,
3:49
and it's called a Preference Dataset.
3:52
In the next lesson, you'll get a chance to take
3:54
a look at one of these datasets in detail, but
3:57
for now, the key takeaway is that the preference dataset indicates
4:00
a human labeler's preference between two possible
4:04
model outputs for the same input.
4:07
Now, it's important to note that this dataset captures the preferences
4:10
of the human labelers, but not human preference in general.
4:14
Creating a preference dataset can be one of
4:17
the trickiest parts of this process, because first you need
4:20
to define your alignment criteria.
4:22
What are you trying to achieve by tuning?
4:24
Do you want to make the model more useful, less toxic, more positive,
4:27
etc?
4:29
You'll need to be clear on this so that you can provide specific
4:31
instructions and choose the correct labelers for the task. But
4:35
once you've done that, step one is complete.
4:37
Next, we move on to step two and we take this preference dataset,
4:41
and we use it to train something called a reward
4:43
model.
4:44
Generally with RLHF and LLMs, this reward model is itself another
4:49
LLM.
4:50
At inference time, we want this reward
4:51
model to take any prompt and a completion
4:54
and return a scalar value that indicates how
4:57
good that completion is for the given prompt.
5:00
So, the reward model is essentially a regression model.
5:03
It outputs numbers.
5:06
The reward model is trained on the preference dataset,
5:08
using the triplets of prompt and two completions,
5:12
the winning candidate and the losing candidate.
5:14
For each candidate completion, we get the model to produce a score,
5:18
and the loss function is a combination of these scores.
5:21
Intuitively, you can think of this as trying to maximize
5:24
the difference in score between the winning candidate and
5:27
the losing candidate.
5:29
And once we've trained this model, we can now pass in a prompt and completion,
5:33
and get back a score indicating how good the completion is.
5:37
The measure of how good a completion is is subjective,
5:40
but you can think of this as the higher the number,
5:42
the better this completion aligns with
5:45
the preferences of the people who labeled the data.
5:48
Once we've completed training this reward model, we'll
5:51
use this model in the final step of this process, where the
5:54
RL of RLHF comes into play. Our goal here is to tune the base large language
5:59
model to produce completions that will maximize the
6:02
reward given by the Reward Model. So, if the
6:05
base LLM produces completions that better
6:07
align with the preferences of the people who labeled
6:10
the data, then it will receive higher rewards from
6:13
the reward model.
6:14
To do this, we introduce a second dataset,
6:17
our prompt dataset.
6:19
This is just, as the name implies, a dataset of prompts,
6:22
no completions.
6:24
Now, before we talk about how this dataset is used, I'm
6:26
going to give you a super quick primer on reinforcement learning. I'm not
6:30
going to go into all the details here, but just
6:32
the key pieces needed to understand the RLHF process at
6:36
a high level. RL is useful when you want to train a model
6:39
to learn how to solve a task that involves a complex and
6:42
fairly open-ended objective.
6:44
You may not know in advance what the optimal solution
6:47
is, but you can give the model rewards to
6:49
guide it towards an optimal series of steps.
6:51
The way we frame problems in reinforcement learning is
6:54
as an agent learning to solve a task by interacting with an
6:58
environment.
6:59
This agent performs actions on the environment, and as a result
7:03
it changes the state of the environment and
7:05
receives a reward that helps it to learn the rules of that
7:08
environment.
7:09
For example, you might have heard about AlphaGo,
7:11
which was a model trained with reinforcement learning.
7:13
It learned the rules for the Board Game
7:15
Go by trying things and receiving rewards or
7:17
penalties based on its actions.
7:20
This loop of taking actions and receiving rewards
7:22
repeats for many steps, and this is how the agent learns.
7:25
Note that this framework differs from supervised learning,
7:28
because there's no supervision.
7:31
The agent isn't shown any examples that map
7:33
from input to output, but instead the agent learns by interacting
7:37
with the environment, exploring a space of possible actions, and
7:40
then adjusting its path.
7:41
The agent's learned understanding of how rewarding each
7:44
possible action is, given the current conditions, are
7:47
saved in a function.
7:49
This function takes as input the current state
7:51
of the environment and outputs a set of
7:53
possible actions that the agent can take next,
7:55
along with the probability that each action will
7:58
lead to a higher reward.
8:00
This function that maps the current state to
8:02
the set of actions is called a Policy, and the goal
8:05
of reinforcement learning is to learn a policy that
8:08
maximizes the reward.
8:10
You'll often hear people describe the policy as the
8:12
brain of the agent, and that's because it's what determines the decisions that the
8:16
agent takes. So now, let's see how these terms relate back to
8:20
reinforcement learning with human feedback.
8:22
In this scenario, the policy is the base large language model
8:26
that we want to tune. The current state is
8:28
whatever is in the context.
8:30
So, something like the prompt and any generated text
8:33
up until this point, and actions are generating tokens.
8:36
Each time the base LLM outputs a completion,
8:39
it receives a reward from the reward model indicating
8:42
how aligned that generated text is.
8:45
Learning the Policy that maximizes the
8:47
reward amounts to a large language model that produces
8:50
completions with high scores from the reward model. Now,
8:53
I'm not going to go into all the details here of
8:55
how this policy is learned, but if you're curious
8:58
to learn a little more, in RLHF, the policy is learned via the
9:03
policy gradient method, proximal policy optimization or PPO. This is
9:07
a standard reinforcement learning algorithm.
9:09
So here's, an overview of everything that happens in each step
9:12
of this process. A prompt is sampled from
9:15
the prompt dataset.
9:17
The prompt is passed to the base large
9:19
language model to produce a completion.
9:22
And this prompt completion pair is passed to
9:24
the reward model to produce a score or reward.
9:27
The weights of the base large language model,
9:30
also known as the policy, are updated via PPO using the reward.
9:34
Each time we update the weights, the policy should get
9:37
a little better at outputting a line text.
9:39
Now, note that I am glossing over a little bit of detail here.
9:42
In practice, you usually add a penalty term to ensure
9:46
the tuned model doesn't stray too far away from the base model,
9:49
but we'll talk a little bit more about that
9:51
in a future lesson.
9:52
This is the high-level picture, but if you want to learn some more detail,
9:55
you can take a look at some of the
9:57
original research papers.
9:58
So, just to recap everything that we've covered, reinforcement
10:01
learning from human feedback is made up of
10:03
three main steps.
10:05
We create a preference data set.
10:07
We use the preference data set to train a reward model.
10:10
And then, we use that reward model in
10:12
a reinforcement learning loop to fine tune our base
10:15
large language model.
10:16
Now, before we get to coding, there's one more detail that's worth
10:19
understanding.
10:20
When it comes to tuning a neural network,
10:22
you might retrain the model by updating all of its weights.
10:26
This is known as full fine-tuning.
10:28
But because large language models are so large,
10:30
updating all of the many weights can take a very long time.
10:34
Instead, we can try out parameter-efficient fine-tuning, which is
10:37
a research area that aims to reduce the
10:39
challenges of fine-tuning large language models by only
10:42
training a small subset of model parameters.
10:45
These parameters might be a subset of the existing model parameters,
10:48
or they could be an entirely new set of parameters.
10:51
Figuring out the optimal methodology is
10:54
an active area of research, but the key benefit here is that you're
10:57
not having to retrain the entire model and
10:59
all of its many weights.
11:01
Parameter Efficient Fine Tuning can also make serving models
11:04
simpler in comparison to fine tuning,
11:06
because instead of having an entirely new model
11:08
that you need to serve, you just use the existing base model,
11:11
and you add on the additional tune parameters.
11:14
You could potentially have one base model with several distinct sets of
11:17
tune parameters that you swap in and out depending on the use
11:21
case or the user that your application is serving.
11:23
So, reinforcement learning from human feedback can
11:26
be implemented with either full fine tuning or
11:28
parameter efficient tuning.
11:30
In this course, when we tune the LLAMA2 Model, we're
11:33
going to be using a parameter efficient implementation.
11:36
This means that the training job won't update all of the base
11:40
large language model weights, only a smaller subset of
11:43
them based on a parameter efficient tuning technique.
11:46
Okay, so now that you know the basics of how RLHF works, let's
11:49
get to coding.
