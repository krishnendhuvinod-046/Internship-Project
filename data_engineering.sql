-- Step 1: Create Table Schema (with snake_case columns)
CREATE TABLE IF NOT EXISTS telecom_data (
    call_failure INTEGER,
    complains INTEGER,
    subscription_length INTEGER,
    charge_amount REAL,
    seconds_of_use INTEGER,
    frequency_of_use INTEGER,
    frequency_of_sms INTEGER,
    distinct_called_numbers INTEGER,
    age_group INTEGER,
    tariff_plan INTEGER,
    status INTEGER,
    age INTEGER,
    customer_value REAL,
    fn REAL,
    fp REAL,
    churn INTEGER
);

-- Step 3: High Value At Risk Feature
-- Logic: Select users where customer_value is above average AND complains = 1
SELECT 
    *,
    CASE 
        WHEN customer_value > (SELECT AVG(customer_value) FROM telecom_data) AND complains = 1 THEN 1 
        ELSE 0 
    END AS high_value_at_risk
FROM telecom_data;
