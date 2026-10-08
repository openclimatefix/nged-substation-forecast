# GB electricity prices: which are free and when each is known

**This page lists the prices that a battery in Great Britain (GB) faces, which of them are free to
download, and when each becomes known.** National Energy System Operator (NESO) is the system
operator. The page is background for the [battery and solar separation
study](../studies/battery-pv-separation.md). How a battery responds to the prices is described on
[how GB batteries schedule themselves](gb-battery-scheduling.md).

**A battery faces several prices. Several free prices are known only after delivery, and three are
known before delivery.** The free prices known before delivery are the N2EX day-ahead price, the
response and reserve auction results, and the bid-offer prices. The table lists the prices that a
battery model can use. [Modo Energy's
explainer](https://modoenergy.com/research/en/wholesale-trading-markets-explainer-gb-n2ex-epex-dayahead-intraday)
describes the day-ahead and intraday markets, and
[Elexon](https://www.elexon.co.uk/settlement/imbalance-pricing/) describes imbalance pricing.

| Price or dataset | Run by, and time resolution | Free to download? | Known when |
|---|---|---|---|
| Day-ahead, N2EX hourly auction ([NESO dataset](https://www.neso.energy/data-portal/day-ahead-power-exchange-prices-nordpool)) | NESO publishes it, hourly | Yes | About 10:00 the day before |
| Day-ahead, EPEX 30-minute and hourly auctions, and intraday auctions ([Nord Pool GB auction](https://support.nordpoolgroup.com/support/solutions/articles/8000088463-about-the-day-ahead-gb-auction)) | EPEX, half-hourly or hourly | No ([EEX webshop](https://webshop.eex-group.com/eex-public-market-data), from €665 a month) | 15:45 the day before, for the 30-minute auction |
| Market index (MID, the EPEX short-term index) | Elexon, half-hourly | Yes | After trading |
| System price and net imbalance volume | Elexon, half-hourly | Yes | Indicative about 30 minutes after the period, revised in later settlement runs |
| Bid-offer prices and acceptances | Elexon, per acceptance | Yes | Bid-offer prices before the period, acceptances during it |
| Response and reserve auction results (Dynamic Containment, Moderation, Regulation, Quick Reserve, and Balancing Reserve; [Enduring Auction Capability (EAC)](https://www.neso.energy/industry-information/balancing-services/enduring-auction-capability-eac)) | NESO, the daily 14:00 auction; [results](https://neso.energy/data-portal/eac-auction-results) | Yes, from November 2023 | The afternoon before delivery |

**The final settled volumes arrive months later.** [Elexon's settlement
runs](https://www.elexon.co.uk/bsc/glossary/1st-reconciliation/) rerun the figures several times.
The Final Reconciliation (RF) run comes 14 months after the day, and a later Dispute Final (DF) run
follows at 28 months. A study of the past can use the final figures, and a forecast made on the day
cannot.

**Day-ahead prices vary widely within the year.** The [Electricity Maps 2025
review](https://www.electricitymaps.com/grid-in-review-2025/great-britain) gives a 2025 day-ahead
mean of £81.2 per MWh, a range of −£30.5 to £958.5, and 190 negative hours (141 in 2024). A negative
price means that producers pay to sell.

**Day-ahead prices are moderately predictable, and imbalance prices are hard to predict.** The
predictability statement is our judgement from the two papers below. [Lago et
al.](https://doi.org/10.1016/j.apenergy.2021.116983) found that the best models in their benchmark
reduce relative mean absolute error to 0.40 to 0.61 of a same-hour-last-week rule, in studies
without GB data (source not checked). [Browell and Gilbert](https://doi.org/10.3390/en15103645)
report that day-ahead forecasts of imbalance prices beat a benchmark by only 3% and intraday
forecasts by 5% to 40%, mostly within 3 hours (source not checked). **A battery's scheduling
software (its optimiser) needs the shape of the day's price curve more than the level.** The shape
is the [price rank](gb-battery-scheduling.md#price-rank) of each half-hour ([Maciejowska et
al.](https://arxiv.org/abs/2511.13616)).
