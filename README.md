### Disclaimer: This is a work in progress. Preliminary results look promising, but analysis is still ongoing. Use at your own risk, but feedback is very welcome.

# Mechanism Learning
## Prior-Free Approximately Incentive Compatible Mechanism Learning
In low trust settings like the internet, situations exist in which individuals are not willing or able to cooperate because of their self-interest. If the goal of the interaction is allocating goods according to preferences and within some given constraints, we can find a solution that maximizes e.g. welfare or revenue despite being in an adversarial setting by using an incentive-compatible (IC) mechanism. This type of mechanism achieves this by making truthful preference revelation the course of action with the highest reward for each individual.

Finding these mechanisms is hard, so for a long time, only solutions for a few specific settings (e.g. [Vickrey auctions](https://en.wikipedia.org/wiki/Vickrey_auction)) were known. Recent advances in computational mechanism design have led to techniques from [machine learning](https://arxiv.org/pdf/1706.03459.pdf) being successfully used to automate the search for mechanisms in a wider range of settings.

This approach is able to learn IC mechanisms from the distributions of the agents' preferences. However, many interesting applications take place in the [prior-free setting](http://www.cs.tau.ac.il/~fiat/mdsem12/amd06.pdf), in which no prior information about agents' preferences is available. The only thing known are the reports the agents submit, but they are free to lie.

Our approach is to pretend the reports are truthful and the agents' distributions have all weight on their report, then use these "distributions" as priors for the mechanism learner. The resulting mechanism will not necessarily be IC, as agents could have lied in their reports to improve their outcome. We should be able to alleviate this by [reducing the sensitivity of the learner using differential privacy](http://kunaltalwar.org/papers/expmech.pdf), bounding the influence a report can have on the outcome of the learning process. Agents could then only achieve bounded gains by misreporting and thus would have a bounded incentive to do so. The resulting mechanism would not be IC, but approximately IC and is expected to yield better results than a lottery.

## Privacy Aware Agents
Since we are also interested in [preserving privacy](https://arxiv.org/pdf/1111.3350.pdf), as a next step we want to consider agents that are averse to making their preferences known to other agents or the computing infrastructure.

Fortunately, the differential privacy used in the above mechanism learner also reduces the amount of information about individual reports that is being leaked from the resulting mechanism. The task here will be to quantify the report privacy the mechanism learner provides and how much value agents can afford to put on privacy while still preserving approximate IC.

We also want to reduce the amount of report information leaked during the learning process. Thus, we can not perform the learning on a central server which would see the plaintext reports of all agents.

On the other hand, we are still interested in guarding against malicious clients, who will try to obtain an outsized influence on the learning process. This means we can not just distribute the computation to the clients, since they could improve their influence on the learning by not properly applying differentially private transformations.

Thus we intend to build a differentially private federated learning backend. The part of the computation that requires knowledge of the reports will be performed client-side in a way that the individual influence on the learning can be verified to be bounded by using range-proofs. The other part will be distributed accross multiple servers that use secure aggregation, never having access to the plaintext reports as long as one of the servers is trustworthy.

# Implementation Details
## Multi-Item Auction Learner
The mechanism learner we base our work on is learning [revenue maximizing multi-item multi-bidder auctions](https://github.com/saisrivatsan/deep-opt-auctions). In this setting, N agents are bidding for M items and revenue for the auctioneer is maximized, all under the constraint that agents shall never pay more for any combination of items than it is worth to them. Reports are the bids for each item. The auction is modelled by an allocation function, determining, given a set of bids, which agents receive which items, and a payment function, determining the payments of each agent to the auctioneer. IC in this setting means that each agent can expect the best allocation and price, under the above constraints, shall happen with the highest likelihood if their bid for each item is exactly what that item is worth to them.

We modify the learner as described above to make it applicable in the prior-free setting, using use [differentially private learning](https://github.com/tensorflow/privacy). Our fork can be found [here](https://github.com/degregat/deep-opt-auctions/). The techniques we develop should be transferable to other classes of mechanisms, e.g. welfare maximizing auctions, trust and reputation systems, or facility placement.

## Distributed Learning Backend
To ensure client privacy we need to distribute the learning process,  while keeping in mind potential data leakage as well as proper execution of the DP learner.

In [Federated Learning](https://github.com/tensorflow/federated), a central server holds a global model, in our case, a pair of functions describing an auction. Each client receives a copy of the global model, derives model updates (in gradient form [link to explanation for gradient]) for it from their reports, and sends the updates back to the server. The server aggregates these updates into the global model to improve it. This process is repeated until the model is sufficiently IC. To make it [differentially private](https://github.com/tensorflow/federated/blob/master/docs/tff_for_research.md#differential-privacy), the gradient updates are clipped (their maximum norm bounded) and noised.

If we want to keep the server from knowing the plaintext submissions, we have to adress three problems:  how to ensure the noising was done properly, how to compute the model update without knowing the plaintext gradients, and how to ensure the clipping was done properly.

#### Ensuring correct noise generation by distributing it accross servers
We cannot trust the users with adding noise, and we need to keep the servers knowledge about the noise limited to prevent inference of gradients. The relief is secure aggregation, a technique based on [secret sharing](https://mortendahl.github.io/2017/06/04/secret-sharing-part1/). We use two aggregation servers for an arbitrary number of clients. Each server generates a part of the noise, splits it into two shares and sends one to the other server. The servers then add their own share and the share they received. This way, no server knows all the noise contained in the sum share.

#### Computing model updates without plaintext gradients using Secure Aggregation
Model updates are again computed using secure aggregation. Each client splits up their gradient into two parts and sends one share to each server. The servers then sum up all the shares they received to compute a sum share. They also add the sum share of the noise obtained from the previous step. Then, the servers together use their sum shares to reconstruct the aggregated gradient. This way, no server ever sees the gradients in plaintext, only the final reconstructed sum, assuming that the servers do not collude.

#### Ensuring correct gradient clipping with Secret Shared Non-Interactive Proofs (SNIPs)
We want to make sure the values clients submit lie within a certain range, but since the servers only receive secret shares, they can not directly check this. With SNIPs, clients can compute a cryptographic range proof and send a share of it along with a share of their submitted gradients, to prove to each server that the norm of the update is correctly bounded.

### Building Blocks for Backend
To implement the above secure aggregation backend with distributed noise generation and the client side SNIPs, we use the following software packages.

#### Prio
[Prio](https://github.com/mozilla/libprio/) is a cryptographic system for secure aggregation with range-proofs on the submissions. Clients are assumed to be malicious, servers to be honest-but-curious. Originally supporting aggregation of boolean vectors, we have extended it to support vectors with fixed-point (approximations of floating point numbers) entries.

#### Google's differential privacy library 
We use Googles implementations of [discrete gaussian noise](https://github.com/google/differential-privacy) to generate the noise on the servers.

## Roadmap
### Mechanism Leaner
- [x] Implement prototype of [Prior-Free Auction Learner](https://github.com/degregat/deep-opt-auctions)
- [ ] Analysis of prototype, regarding approximate IC
- [ ] Analysis of prototype in privacy aware setting
- [ ] Port mechanism learner to finished backend

### Backend
- [x] Implement integer support for libprio
- [x] Implement fixed-point support for libprio
- [ ] Implement secure distributed noise generator with secure aggregation
- [ ] Integrate distributed noise generator with federated learning framework
