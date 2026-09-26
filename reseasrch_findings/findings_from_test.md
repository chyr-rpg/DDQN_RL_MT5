# What the Testing Results Changed in How I Look at the System

When I first started testing DDQN-RL-MT5, I naturally focused on the final result of each run: return, profit factor, Sharpe ratio, drawdown, and whether the balance curve looked reasonable.

After testing the system across different markets, I gradually realised that the final profit alone was not telling me enough.

The question I became more interested in was:

> **What did the system have to go through to produce that profit?**

This matters especially because the current execution framework still contains basket and averaging behaviour.

A basket can eventually close with a profit, but that does not necessarily mean the process was efficient. It may have spent a long period in floating loss, added several positions, or taken much more exposure than the final realised result suggests.

This became one of the more important findings from the project so far.

## Profit alone can hide very different behaviour

The tests so far have produced quite different trading profiles.

EURUSD M5 generated a historical return of **47.27%**, with **12.36% equity drawdown**, a profit factor of **1.91**, and 335 trades.

SP500 M15 was much less active. It generated **3.46% return**, **3.54% equity drawdown**, a profit factor of **2.25**, and only 52 trades.

XAUUSD M1 was the most active test, with 1,788 trades. It also produced the highest historical return at **154.99%**, but experienced the highest equity drawdown among these tests at **17.87%**.

The latest GBPUSD M1 test showed another type of behaviour. From **1 September to 24 September 2026**, it generated **273 trades**, a historical return of approximately **3.57%**, a profit factor of **1.90**, and **1.90% maximum equity drawdown**.

Its Sharpe ratio was **4.15**, although I would be careful about reading too much into that number because this particular test covers a relatively short period.

Looking only at headline performance, it would be easy to focus on XAUUSD because its return was much higher.

I do not think that is the most useful way to interpret these results.

These tests use different markets, timeframes, testing periods and asset-specific parameters. They are therefore not designed as a controlled comparison of which instrument is "better".

What they do show is that the same learning architecture can behave very differently when placed in different market environments.

More importantly, they made me realise that a profitable outcome and a good decision are not necessarily the same thing.

## A profitable recovery can still be a poor decision

Consider two baskets.

The first opens a position, moves slightly against it, adds one additional trade, and then recovers relatively quickly.

The second also ends with a profit, but only after building a much deeper basket, remaining underwater for a long period, and taking substantially more exposure.

If the reinforcement-learning reward mainly looks at the final realised P/L, both experiences may appear successful.

From a risk perspective, however, they are clearly not the same.

The latest GBPUSD test gave me a useful example of this difference.

Its **maximum balance drawdown was only 0.32%**, while maximum equity drawdown reached **1.90%**.

The numbers themselves are not particularly large, but the gap between them is important.

If I only looked at the balance curve, the test would appear extremely smooth. Equity tells a different story because it also captures what was happening inside the open positions before those profits or losses were realised.

This is why I started paying much more attention to **equity drawdown**, rather than relying on balance drawdown alone.

Balance reflects what has already been realised. Equity also reflects what the system is still carrying.

For a basket-based strategy, that distinction matters.

A relatively smooth balance curve can still hide considerably more floating stress underneath it.

Once I started reading the results this way, the weakness of a simple reward structure became much clearer.

The learning objective should not simply be:

> profit = good, loss = bad.

The agent needs some way to understand the **quality of the path** that produced the result.

## This changed how I designed the reward and memory system

As the project developed, the objective gradually shifted from rewarding profitable trades toward rewarding **better-quality trading behaviour**.

The system therefore started considering more information around each episode, including basket depth, floating drawdown, recovery quality, additional entries, danger states, and previous drawdown events.

I also became more interested in separating different types of experience.

For example, an episode that earns a modest profit using one or two positions should probably not be remembered in exactly the same way as an episode that earns the same profit only after a large basket expansion.

Both episodes contain useful information.

But the lesson behind each one is different.

This was one of the reasons the architecture gradually developed separate replay and memory components for recent experience, dangerous periods, deep baskets, efficient periods, persistent Q-memory, and drawdown events.

The purpose is not simply to give the neural network more historical data.

It is to give the system a better way to distinguish **what kind of experience it is learning from**.

## Different markets exposed different weaknesses

Testing across several markets also changed how I look at cross-asset backtesting.

XAUUSD generated far more decisions and therefore many more learning events. This made it useful for observing how the policy and memory components evolve when the agent receives frequent feedback.

EURUSD contained periods where basket recovery and drawdown behaviour became much more important. Those periods were particularly useful when I started thinking about danger memory and risk-sensitive reward design.

SP500 created almost the opposite problem.

With only 52 trades in the test, the agent receives much less feedback. That raises another question: how much confidence should I place in behaviour learned from a relatively small number of observations?

GBPUSD added another useful angle.

It generated 273 trades in less than one month and maintained relatively low overall drawdown. At the same time, the difference between its **0.32% balance drawdown** and **1.90% equity drawdown** again showed why realised results alone can miss part of the behaviour happening inside a basket.

It also reminded me that apparently strong statistics need context.

For example, a Sharpe ratio of 4.15 looks very strong at first glance, but this test only covers around three weeks of September data. I would rather see whether similar behaviour continues across a much longer and unseen period before treating that statistic as meaningful evidence.

So rather than asking:

> Which market produced the best backtest?

I now find it more useful to ask:

> **What did each market reveal about the architecture?**

For me, that is a much more useful way to compare the testing results.

## One important limitation remains

There is also an important limitation in the current results.

These tests were generated with:

```text
TrainingMode = true
```

The latest GBPUSD run used the same setting.

This means the agent continued updating its model and memory while moving through the historical data, which is a strength for real-life trading.

However, this also makes me not treat the current results as strict out-of-sample evidence.

They show how the system behaved **while learning from the same historical period it was being evaluated on**.

That is useful for studying the learning process, debugging the architecture, and understanding how different memory and reward mechanisms behave.

But it is different from testing whether an already-trained policy can generalise to unseen data.

The next step is therefore to separate these two stages more clearly:

```text
Training period
        ↓
Agent learns
        ↓
Save model and memory
        ↓
Later unseen period
        ↓
Freeze learning
        ↓
Evaluate the policy
```

That should make the next round of testing much more meaningful.

Instead of asking whether the agent can learn its way through a historical period, the question becomes whether what it learned can still produce sensible behaviour when the market data is new.

For me, this is probably the most important transition in the project now.

The tests so far have been useful for finding weaknesses in the architecture and understanding how the agent behaves under different market conditions.

The next stage is less about producing a better-looking backtest.

It is about finding out whether the behaviour the agent learned can actually survive once it stops learning from the test itself.
