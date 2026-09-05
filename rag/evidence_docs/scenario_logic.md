# Scenario Logic

Scenario analysis is used to answer planning questions that model accuracy alone cannot answer.

The scenario engine tests how demand changes when assumptions change.

## Scenario Flow

assumption change → patient opportunity → access → competition adjustment → persistence → supply → forecasted demand

## Scenarios Included

1. Base case  
No change from the governed baseline forecast.

2. Access downside  
Access falls by 15 percent.

3. Earlier strong competitor pressure  
Overlapping competition increases for biomarker-positive therapies A, B, and D.

4. Epidemiology upside  
Eligible patient pool increases by 12 percent.

5. Persistence downside  
Average persistence worsens, reducing active treated demand by 8 percent.

6. Therapy D East supply constraint  
Therapy D in East has a 70 percent supply fill rate.

7. Combined downside  
Access falls, competitor pressure increases, persistence worsens, and Therapy D East supply is constrained.

## Important Boundary

The numerical scenario engine calculates the official values. The RAG and LLM layer may explain these values, but must not invent, alter, or recalculate them.
