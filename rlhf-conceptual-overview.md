RLHF is a technique we can use to try and better align an LLM's output with user intention and preference. 

Here's a conceptual overview of RLHF.

Let's say that we want to tune a model on a summarization task.
<img width="1916" height="959" alt="picture 1" src="https://github.com/user-attachments/assets/c9b413ad-a076-433a-bae7-816c4887ae7a" />

We might start by gathering some text samples to summarize and then have humans produce a summary for each input. So for example, here we have the input text, 

""before I go to university, I want to take a road trip in Europe. I've lived in several European cities, but there's still a lot I haven't seen, etc.""

And then, we have a corresponding summary of that text. 

""The user wants to take a road trip in Europe before university. They want to see as much as possible in a short time, and they're wondering if they should go to places that are significant from their childhood or places they have never seen.""

We can use these human-generated summaries to create pairs of input text and summary, and we could train a model directly on a bunch of these pairs. But the thing is, there's no one correct way to summarize a piece of text. Natural language is flexible, and there are often many ways to say the same thing. 
<img width="1915" height="957" alt="picture 2" src="https://github.com/user-attachments/assets/db017deb-2fe7-452d-a6cd-538dad11a0ec" />

For example, here's an equally valid summary. And in fact, there are many more valid summaries we could write. Each summary might be technically correct, but different people, different groups of people, different audiences will all have a preference. And preferences are hard to quantify. Some problems like entity extraction or classification have correct answers, but sometimes the task we want to teach the model doesn't have a clear objective best answer.
<img width="1908" height="952" alt="picture 3" src="https://github.com/user-attachments/assets/4ea76861-2297-498d-bf7c-0d62b5ee838a" />

So, instead of trying to find the best summary for a particular piece of input text, we're gonna frame this problem a little differently. 

We're going to gather information on human preferences, and to do that, we'll provide a human labeler with two candidate summaries and ask the labeler to pick which one they prefer. And instead of the standard supervised tuning process where we tune the model to map an input to a single correct answer, we'll use reinforcement learning to tune the model to map an input to a single correct answer, we'll use reinforcement learning to tune the model to produce responses that are aligned with human preferences. 

So how does all this work? Well, it's an evolving area of research and there are a lot of variations and how we might implement RLHF specifically, but the high level themes are the same.
<img width="1910" height="958" alt="picture 4" src="https://github.com/user-attachments/assets/ef2b8de7-cce6-4f8a-965a-78b2f1154528" />

RLHF consists of three stages. 

* First, we create a preference data set.
* Then, we use this preference data set to train a reward model with supervised learning.
* And then, we use the reward model in a reinforcement learning loop to fine tune our base large language model.

Let's look at each of these steps in detail.
<img width="1908" height="958" alt="picture 5" src="https://github.com/user-attachments/assets/26add048-ffee-418b-b7d0-d139b62477b8" />

First things first, we're going to start with the large language model that we want to tune. In other words, the base LLM.

In L2-L5 we tune the open source LLMA2 Model. But before we actually do any model tuning, we're going to use this base LLM to generate completions for a set of prompts. 

So for example, we might send the input prompt, summarize the following text, I want to start gardening, but et cetera. And we would get the model to generate multiple output completions for the same prompt. And then, we have human labelers rate these completions.

Now, the first way you might think to do this is to have the human labelers indicate on some absolute scale how good the completion is. But this doesn't yield the best results in practice because scales like this are subjective and they tend to vary across people.
<img width="1910" height="959" alt="picture 6" src="https://github.com/user-attachments/assets/f7dd70ce-c3b7-481a-bb23-5829d68d29bc" />

Instead, one way of doing this that's worked pretty well is to have the human labeler compare two different output completions for the same input prompt, and then specify which one they prefer. This is the dataset that I referenced earlier, and it's called a Preference Dataset.

The key takeaway is that the preference dataset indicates a human labeler's preference between two possible model outputs for the same input. Now, it's important to note that this dataset captures the preferences of the human labelers, but not human preference in general. Creating a preference dataset can be one of the trickiest parts of this process, because first you need to define your alignment criteria.

* What are you trying to achieve by tuning?
* Do you want to make the model more useful, less toxic, more positive, etc?

You'll need to be clear on this so that you can provide specific instructions and choose the correct labelers for the task. 
Once you've done that, step one is complete.
<img width="1912" height="956" alt="picture 7" src="https://github.com/user-attachments/assets/d1d220cb-ee17-47ab-aa31-e56dd65fa720" />

Next, we move on to step two and we take this preference dataset, and we use it to train something called a reward model. Generally with RLHF and LLMs, this reward model is itself another LLM. At inference time, we want this reward model to take any prompt and a completion and return a scalar value that indicates how good that completion is for the given prompt. So, the reward model is essentially a regression model. It outputs numbers.
<img width="1910" height="951" alt="picture 8" src="https://github.com/user-attachments/assets/b603d17c-ab75-4c74-af14-034c994c94d8" />

The reward model is trained on the preference dataset, using the triplets of prompt and two completions, the winning candidate and the losing candidate. For each candidate completion, we get the model to produce a score, and the loss function is a combination of these scores. Intuitively, you can think of this as trying to maximize the difference in score between the winning candidate and the losing candidate.
<img width="1912" height="956" alt="picture 9" src="https://github.com/user-attachments/assets/e08f1020-d5b5-483e-a24a-2d4d7e6792bd" />

Once we've trained this model, we can now pass in a prompt and completion, and get back a score indicating how good the completion is. The measure of how good a completion is is subjective, but you can think of this as the higher the number,the better this completion aligns with the preferences of the people who labeled the data.
<img width="1910" height="956" alt="picture 10" src="https://github.com/user-attachments/assets/99e4cecd-fb57-41e8-b63d-1e6f4c7ca97a" />

Once we've completed training this reward model, we'll use this model in the final step of this process, where the RL of RLHF comes into play. Our goal here is to tune the base large language model to produce completions that will maximize the
reward given by the Reward Model. If the base LLM produces completions that better align with the preferences of the people who labeled the data, then it will receive higher rewards from the reward model.

To do this, we introduce a second dataset, our prompt dataset. This is just, as the name implies, a dataset of prompts, no completions.

Now, I'm going to give you a super quick primer on reinforcement learning. Just the key pieces needed to understand the RLHF process at

a high level. RL is useful when you want to train a model to learn how to solve a task that involves a complex and fairly open-ended objective. You may not know in advance what the optimal solution is, but you can give the model rewards to guide it towards an optimal series of steps.
<img width="1912" height="955" alt="picture 11" src="https://github.com/user-attachments/assets/7587194a-c085-434c-85f1-db3352e77e0d" />

The way we frame problems in reinforcement learning is as an agent learning to solve a task by interacting with an environment. This agent performs actions on the environment, and as a result it changes the state of the environment and receives a reward that helps it to learn the rules of that environment.

For example, you might have heard about AlphaGo, which was a model trained with reinforcement learning. It learned the rules for the Board Game Go by trying things and receiving rewards or penalties based on its actions. This loop of taking actions and receiving rewards repeats for many steps, and this is how the agent learns. Note that this framework differs from supervised learning, because there's no supervision.

The agent isn't shown any examples that map from input to output, but instead the agent learns by interacting with the environment, exploring a space of possible actions, and then adjusting its path. The agent's learned understanding of how rewarding each possible action is, given the current conditions, are saved in a function.

This function takes as input the current state of the environment and outputs a set of possible actions that the agent can take next, along with the probability that each action will lead to a higher reward.
<img width="1910" height="950" alt="picture 12" src="https://github.com/user-attachments/assets/72c25914-8275-4178-8c5a-5932a2420eb4" />

This function that maps the current state to the set of actions is called a Policy, and the goal of reinforcement learning is to learn a policy that maximizes the reward. You'll often hear people describe the policy as the brain of the agent, and that's because it's what determines the decisions that the agent takes. So now, let's see how these terms relate back to reinforcement learning with human feedback.
<img width="1911" height="957" alt="picture 13" src="https://github.com/user-attachments/assets/5a247a42-2f27-424b-b44f-0d56142d196a" />

In this scenario, the policy is the base large language model that we want to tune. The current state is whatever is in the context. So, something like the prompt and any generated text up until this point, and actions are generating tokens. Each time the base LLM outputs a completion, it receives a reward from the reward model indicating how aligned that generated text is.

Learning the Policy that maximizes the reward amounts to a large language model that produces completions with high scores from the reward model. Now, I'm not going to go into all the details here of how this policy is learned, but if you're curious to learn a little more, in RLHF, the policy is learned via the policy gradient method, proximal policy optimization or PPO. This is a standard reinforcement learning algorithm.
<img width="1914" height="957" alt="picture 14" src="https://github.com/user-attachments/assets/ecb70d1b-5e6d-43cf-8bfa-6ee81d9bcba2" />

Here's, an overview of everything that happens in each step of this process. 

* A prompt is sampled from the prompt dataset.
* The prompt is passed to the base large language model to produce a completion.
* And this prompt completion pair is passed to the reward model to produce a score or reward.
* The weights of the base large language model, also known as the policy, are updated via PPO using the reward.
* Each time we update the weights, the policy should get a little better at outputting a line text.

Note, this is high-level, in practice, you usually add a penalty term to ensure the tuned model doesn't stray too far away from the base model. This is the high-level picture.
<img width="1914" height="958" alt="picture 15" src="https://github.com/user-attachments/assets/bf5e875d-77aa-4bad-bb08-204bf9017e45" />

If you want to learn some more detail, you can take a look at some of the original research papers.
<img width="1912" height="951" alt="picture 16" src="https://github.com/user-attachments/assets/e6f23e78-a236-4866-8c42-27416b1877b2" />

To recap: 

Reinforcement learning from human feedback is made up of three main steps.
* We create a preference data set.
* We use the preference data set to train a reward model.
* And then, we use that reward model in a reinforcement learning loop to fine tune our base large language model.
<img width="1911" height="951" alt="picture 17" src="https://github.com/user-attachments/assets/4601859a-3f3a-4b61-a66f-5f5723c5390d" />

Here's one more detail that's worth understanding.

When it comes to tuning a neural network, you might retrain the model by updating all of its weights. This is known as full fine-tuning. But because large language models are so large, updating all of the many weights can take a very long time. Instead, we can try out parameter-efficient fine-tuning, which is a research area that aims to reduce the challenges of fine-tuning large language models by only training a small subset of model parameters.

These parameters might be a subset of the existing model parameters, or they could be an entirely new set of parameters.

Figuring out the optimal methodology is an active area of research, but the key benefit here is that you're not having to retrain the entire model and all of its many weights.

Parameter Efficient Fine Tuning can also make serving models simpler in comparison to fine tuning, because instead of having an entirely new model that you need to serve, you just use the existing base model, and you add on the additional tune parameters.

You could potentially have one base model with several distinct sets of tune parameters that you swap in and out depending on the use case or the user that your application is serving. So, reinforcement learning from human feedback can be implemented with either full fine tuning or parameter efficient tuning.

In L2-L5 we tune the LLAMA2 Model, we're going to be using a parameter efficient implementation.

This means that the training job won't update all of the base large language model weights, only a smaller subset of them based on a parameter efficient tuning technique.

Source:
1. DeepLearnin ai reinforement learning lab
