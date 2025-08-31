SELECT
DISTINCT(race_id) as race_id,
course as race_course,
race_type,
going as race_going_condition,
direction as race_direction,
meeting_date as date,
distance_yards,
winning_time_secs,
added_money
FROM
historic_races
WHERE (
    race_id>0
    AND
    winning_time_secs>:min_winner_time
    AND
    (distance_yards/winning_time_secs)>:min_winner_speed
    AND
    (distance_yards/winning_time_secs)<:max_winner_speed
    AND
    going IS NOT NULL
    AND
    direction IS NOT NULL   
    AND
    added_money IS NOT NULL
    AND
    added_money>:min_total_prize     
)
ORDER BY
race_id;
