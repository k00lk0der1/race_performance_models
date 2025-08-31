SELECT
runner_id,
race_id,
gender as runner_gender,
age as runner_age,
bred as runner_breeding_country,
distance_travelled,
distance_behind_winner,
finish_position
FROM 
historic_runners 
WHERE (
    race_id IN (
        SELECT 
        race_id 
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
    ) AND (
        age>:min_runner_age AND age<:max_runner_age
    )
);
