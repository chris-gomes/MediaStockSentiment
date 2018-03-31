SELECT
  g.MonthYear AS MonthYear,
  g.Actor1Name AS Actor1Name,
  g.Actor2Name AS Actor2Name,
  AVG(g.GoldsteinScale) AS GoldsteinScale,
  AVG(g.NumMentions) AS NumMentions,
  AVG(g.NumSources) AS NumSources,
  AVG(g.AvgTone) AS AvgTone
FROM
  [gdelt-bq:full.events] g
WHERE
  ActionGeo_CountryCode = "US"
GROUP BY
  MonthYear,
  Actor1Name,
  Actor2Name
