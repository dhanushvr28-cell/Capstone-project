-- Part A, task 6: seven business questions against data/instamart.db
-- Each query starts with a line "-- name: Qn" so src/part_a_sql.py can split and run them one by one.
-- Unless stated otherwise only completed orders count as revenue or spend. All amounts are INR.

-- name: Q1
-- WHERE filters rows before grouping (completed orders above INR 500);
-- HAVING filters groups after aggregation (customers whose total of those orders exceeds INR 8000).
SELECT customer_id,
       COUNT(*)                    AS big_orders,
       ROUND(SUM(order_amount), 2) AS total_big_order_amount
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_big_order_amount DESC;

-- name: Q2
-- Three-table join: completed revenue by city and product category.
SELECT c.city,
       p.category,
       ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products  p ON p.product_id  = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- name: Q3
-- Scalar subquery: products priced above the average catalogue price.
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- name: Q4
-- LEFT JOIN: customers who have no completed order (the join condition carries the status filter,
-- so customers with only cancelled orders are kept on the left side with NULLs on the right).
SELECT c.customer_id, c.city, c.signup_date
FROM customers c
LEFT JOIN orders o
       ON o.customer_id = c.customer_id
      AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- name: Q5
-- Two CTEs: per-customer completed spend, then the average of those totals.
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) AS avg_total FROM customer_spend
)
SELECT cs.customer_id,
       ROUND(cs.total_spend, 2) AS total_spend,
       ROUND(a.avg_total, 2)    AS avg_total_spend
FROM customer_spend cs
CROSS JOIN avg_spend a
WHERE cs.total_spend > a.avg_total
ORDER BY cs.total_spend DESC;

-- name: Q6
-- Ranking window functions within each city by total completed spend.
WITH spend AS (
    SELECT o.customer_id, c.city, ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY o.customer_id, c.city
)
SELECT city,
       customer_id,
       total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC, customer_id) AS row_num
FROM spend
ORDER BY city, row_num;

-- name: Q7
-- Running total: daily completed revenue and its cumulative sum.
WITH daily AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day,
           SUM(order_amount)       AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day,
       ROUND(revenue, 2)                                AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2)       AS cumulative_revenue
FROM daily
ORDER BY day;
