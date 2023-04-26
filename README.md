### Disclaimer: This is a work in progress. Preliminary results look promising, but analysis is still ongoing. Use at your own risk, but feedback is very welcome.

# Mechanism Learning
## Prior-Free Approximately Incentive Compatible Mechanism Learning
In low trust settings like the internet, situations exist in which individuals are not willing or able to cooperate because of their self-interest. If the goal of the interaction is allocating goods according to preferences and within some given constraints, we can find a solution that maximizes e.g. welfare or revenue despite being in an adversarial setting by using an incentive-compatible (IC) mechanism. This type of mechanism achieves this by making truthful preference revelation the course of action with the highest reward for each individual.

Finding these mechanisms is hard, so for a long time, only solutions for a few specific settings (e.g. [Vickrey auctions](https://en.wikipedia.org/wiki/Vickrey_auction)) were known. Recent advances in computational mechanism design have led to techniques from [machine learning](https://arxiv.org/pdf/1706.03459.pdf) being successfully used to automate the search for mechanisms in a wider range of settings.

This approach is able to learn IC mechanisms from the distributions of the agents' preferences. However, many interesting applications take place in the [prior-free setting](http://www.cs.tau.ac.il/~fiat/mdsem12/amd06.pdf), in which no prior information about agents' preferences is available. The only thing known are the reports the agents submit, but they are free to lie.

Our approach is to pretend the reports are truthful and the agents' distributions have all weight on their report, then use these "distributions" as priors for the mechanism learner. The resulting mechanism will not necessarily be IC, as agents could have lied in their reports to improve their outcome. We should be able to alleviate this by [reducing the sensitivity of the learner using differential privacy](http://kunaltalwar.org/papers/expmech.pdf), bounding the influence a report can have on the outcome of the learning process. Agents could then only achieve bounded gains by misreporting and thus would have a bounded incentive to do so. The resulting mechanism would not be IC, but approximately IC and is expected to yield better results than a lottery.

### Next Step: Online Learning for the Prior-Independent Case
If we assume some distribution of preferences exists, but we don't know it, we can use the online version of [two player auction learning](https://github.com/degregat/two-player-auctions). In this case, every set of bids is used as a learning sample, to approximate the distribution of preferences over time. Since agents can still misreport, [input perturbation](https://arxiv.org/abs/2002.08570) (a specific differential privacy setup) should be used to bound the per sample influence on the learning of the distribution. This way, we do not have to learn a single auction per bid sample, but can reuse learned information about the preference space. This is not as general as the prior-free case, but can be used for iterated use cases where agents preferences follow recurring patterns.

### Long Term: Mechanism Learning with unknown preferences
For all of the above to learn, Agents need to know their preferences. If they don't they can be provided with a [feedback mechanism](https://arxiv.org/abs/2004.08924) to learn them from mechanism outcomes.

## Privacy Aware Agents
Since we are also interested in [preserving privacy](https://arxiv.org/pdf/1111.3350.pdf), as a next step we want to consider agents that are averse to making their preferences known to other agents or the computing infrastructure.

Fortunately, the differential privacy used in the above approaches also reduces the amount of information about individual reports that is being leaked from the resulting mechanism. The task here will be to quantify the report privacy the mechanism learner provides and how much value agents can afford to put on privacy while still preserving approximate IC.

We also want to reduce the amount of report information leaked during the bid aggregation process. Thus, we can not perform the aggregation on a central server which would see the plaintext reports of all agents.

On the other hand, we are still interested in guarding against malicious clients, who will try to obtain an outsized influence on the learning process. This means we can not just distribute the computation to the clients, since they could improve their influence on the learning by not properly applying differentially private transformations.

Thus we are building a differentially private secure aggregation backend. The part of the computation that requires knowledge of the reports will be performed client-side in a way that the individual influence on the mechanism can be verified to be bounded by using range-proofs. The rest will be distributed accross multiple servers that use secure aggregation, never having access to the plaintext reports as long as one of the servers is trustworthy.

# Implementation Details
## Multi-Item Auction Learner
The mechanism learner we base our work on is learning [revenue maximizing multi-item multi-bidder auctions](https://github.com/saisrivatsan/deep-opt-auctions). In this setting, N agents are bidding for M items and revenue for the auctioneer is maximized, all under the constraint that agents shall never pay more for any combination of items than it is worth to them. Reports are the bids for each item. The auction is modelled by an allocation function, determining, given a set of bids, which agents receive which items, and a payment function, determining the payments of each agent to the auctioneer. IC in this setting means that each agent can expect the best allocation and price, under the above constraints, shall happen with the highest likelihood if their bid for each item is exactly what that item is worth to them.

We modify the learners as described above to make them applicable in the prior-free and prior-independent (WIP) settings, using [differentially private learning](https://github.com/tensorflow/privacy). Our forks can be found [here](https://github.com/degregat/deep-opt-auctions/) and [here](https://github.com/degregat/two-player-auctions). The techniques we develop should be transferable to other classes of mechanisms, e.g. welfare maximizing auctions, trust and reputation systems, or facility placement.

## Distributed Learning Backend
To achieve bid aggregation which is private and robust against misreporting, as well as guarantee the correct execution of the learning process, we need to do the following: 

#### Ensure correct bid clipping with Secret Shared Non-Interactive Proofs (SNIPs)
We want to make sure the values clients submit lie within a certain range, but since the servers only receive secret shares, they can not directly check this. With SNIPs, clients can compute a cryptographic range proof and send a share of it along with a share of their submitted inputs, to prove to each server that the inputs are within the permissible range.

#### Ensure correct noise generation by distributing it accross servers
We cannot trust the users with adding noise, since they lose influence in noisier settings, and we need to keep the servers knowledge about the noise limited to prevent inference of submissions. The relief is secure aggregation, a technique based on [secret sharing](https://mortendahl.github.io/2017/06/04/secret-sharing-part1/). We use two aggregation servers for an arbitrary number of clients. Each server generates a part of the noise, scales it and adds it to the submission shares. This way, no server knows all the noise contained in the reconstructed sum. The servers are assumed to be honest-but-curious.

#### Verifiable Auction Learning
After aggregating the bids and perturbing them, they can be used as inputs for online learning with the [two player auction learning](https://github.com/degregat/two-player-auctions). To prevent a single server from being able skew the model, we need to either robustly distribute it, or make the learning verifiable.

##### Open Problem: Practically Verifiable Learning
Since in our use-case the bids are (tuneably) private after aggregation and input perturbation, we can work towards building a publicly verifiable mechanism learning process, without needing to use expensive cryptographic techniques:
Using shared public randomness (e.g. a beacon, blockchain, etc.) and [deterministic](https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html) machine learning frameworks, we can compute the model on a couple of servers and run a quorum to see if they come up with the same results to raise the cost of compromise.

### Building Blocks for Backend
To implement the above secure aggregation backend with distributed noise generation and the client side SNIPs, we use the following software packages.

#### Prio
[Prio](https://github.com/mozilla/libprio/) is a cryptographic system for secure aggregation with range-proofs on the submissions. Clients are assumed to be malicious, servers to be honest-but-curious. Originally supporting aggregation of boolean vectors, we have extended it to support vectors with fixed-point (approximations of floating point numbers) entries.

#### Google's differential privacy library
We use Googles implementations of [discrete gaussian noise](https://github.com/google/differential-privacy) to generate the noise on the servers.

### Next Step: Move Backend Prototype closer to Production Quality
An effort to implement a production version of the the backend is currently under way with the [dpsa project](https://github.com/dpsa-project/overview).

## Roadmap
### Mechanism Leaner
- [x] Implement prototype of [Prior-Free Auction Learner](https://github.com/degregat/one-shot-approx-auctions)
- [x] Analysis of prototype, regarding approximate IC ([arXiv](https://arxiv.org/abs/2104.00159)) ([ICLR DPML Workshop](https://dp-ml.github.io/2021-workshop-ICLR/files/27.pdf))
- [x] Implement [Two-Player Auction Learning in JAX](https://github.com/degregat/two-player-auctions) ([intro blogpost](https://iclr-blog-track.github.io/2022/03/25/two-player-auction-learning/)) for more efficient experimentation in the prior independent case.
  - [ ] Add [Sacred](https://github.com/IDSIA/sacred) support for reproducible experimentation
  - [ ] Add misreporting for individual agents
  - [ ] Add Differentially Private bid elicitation with input perturbation
- [ ] Port Two-Player Auction Learner to finished backend

### Backend
- [x] Implement integer support for libprio
- [x] Implement fixed-point support for libprio
- [x] Implement secure distributed noise generator with secure aggregation ([code](https://github.com/degregat/prio-dp))
- [ ] Production version of backend
  - See [roadmap](https://github.com/dpsa-project/overview#roadmap) over at dpsa project.
