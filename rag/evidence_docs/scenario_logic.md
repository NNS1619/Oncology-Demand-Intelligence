Scenario Logic

Scenario analysis answers planning questions that model accuracy alone cannot answer. The scenario engine tests how demand changes when assumptions change.

Scenario Flow

assumption change -> patient opportunity -> access -> competition adjustment -> persistence -> supply -> forecasted demand

Scenarios Included

Base case: no change from the governed baseline forecast.

Access downside: access falls by 15 percent.

Earlier strong competitor pressure: overlapping competition increases for biomarker-positive therapies A, B, and D.

Epidemiology upside: eligible patient pool increases by 12 percent.

Persistence downside: average persistence worsens, reducing active treated demand by 8 percent.

Therapy D East supply constraint: Therapy D in East has a 70 percent supply fill rate.

Combined downside: access falls, competitor pressure increases, persistence worsens, and Therapy D East supply is constrained.

Important Boundary

The numerical scenario engine calculates the official values. The RAG and LLM layer may explain those values but must not invent, alter, or recalculate them.

The P10, P50, and P90 outputs are simulated planning quantiles. P10 is a conservative
planning case, P50 is the median central planning case, and P90 is an upside
planning case. They describe uncertainty under the synthetic assumptions and Monte
Carlo design. They are not clinical confidence intervals, prediction guarantees, or
evidence that the assumed input distributions are correct.

P50 should not be described as the most likely value unless the simulated distribution's
mode has been estimated separately. A median and a mode are different statistical ideas.
