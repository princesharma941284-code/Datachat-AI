"""50 benchmark questions with a trusted ("gold") DuckDB query for each.

Tables: customers, products, orders   (see sample_data.py)
"revenue" always means SUM(orders.total_amount).

ordered=True  -> the order of the rows matters (top-N / ranking questions)
ordered=False -> same rows in any order count as correct
"""

CATEGORIES = ["Aggregation", "Filtering", "Group by", "Join", "Date & time", "Ranking"]


def _q(category, difficulty, question, gold_sql, ordered=False):
    return {
        "category": category,
        "difficulty": difficulty,
        "question": question,
        "gold_sql": gold_sql.strip(),
        "ordered": ordered,
    }


_RAW = [
    # ---------------- Aggregation (8) ----------------
    _q("Aggregation", "easy", "How many orders are there in total?",
       "SELECT COUNT(*) FROM orders"),
    _q("Aggregation", "easy", "What is the total revenue?",
       "SELECT SUM(total_amount) FROM orders"),
    _q("Aggregation", "easy", "What is the average order amount?",
       "SELECT AVG(total_amount) FROM orders"),
    _q("Aggregation", "easy", "What is the highest unit price among all products?",
       "SELECT MAX(unit_price) FROM products"),
    _q("Aggregation", "easy", "How many customers do we have?",
       "SELECT COUNT(*) FROM customers"),
    _q("Aggregation", "easy", "In how many different cities do our customers live?",
       "SELECT COUNT(DISTINCT city) FROM customers"),
    _q("Aggregation", "easy", "What is the total quantity of items ordered?",
       "SELECT SUM(quantity) FROM orders"),
    _q("Aggregation", "easy", "What are the smallest and the largest order amounts?",
       "SELECT MIN(total_amount), MAX(total_amount) FROM orders"),

    # ---------------- Filtering (8) ----------------
    _q("Filtering", "easy", "How many orders were cancelled?",
       "SELECT COUNT(*) FROM orders WHERE status = 'Cancelled'"),
    _q("Filtering", "easy", "How many orders have a discount of more than 20 percent?",
       "SELECT COUNT(*) FROM orders WHERE discount > 0.2"),
    _q("Filtering", "easy", "List the names of all customers who live in Delhi.",
       "SELECT customer_name FROM customers WHERE city = 'Delhi'"),
    _q("Filtering", "easy", "How many products cost more than 5000?",
       "SELECT COUNT(*) FROM products WHERE unit_price > 5000"),
    _q("Filtering", "medium", "What is the total revenue of delivered orders only?",
       "SELECT SUM(total_amount) FROM orders WHERE status = 'Delivered'"),
    _q("Filtering", "easy", "How many orders have a quantity of 5 or more?",
       "SELECT COUNT(*) FROM orders WHERE quantity >= 5"),
    _q("Filtering", "easy", "How many customers belong to the Corporate segment?",
       "SELECT COUNT(*) FROM customers WHERE segment = 'Corporate'"),
    _q("Filtering", "medium", "How many orders were either returned or cancelled?",
       "SELECT COUNT(*) FROM orders WHERE status IN ('Returned', 'Cancelled')"),

    # ---------------- Group by (8) ----------------
    _q("Group by", "medium", "How many customers live in each city?",
       "SELECT city, COUNT(*) FROM customers GROUP BY city"),
    _q("Group by", "medium", "What is the total revenue for each order status?",
       "SELECT status, SUM(total_amount) FROM orders GROUP BY status"),
    _q("Group by", "medium", "How many products are there in each category?",
       "SELECT category, COUNT(*) FROM products GROUP BY category"),
    _q("Group by", "medium", "What is the average unit price in each product category?",
       "SELECT category, AVG(unit_price) FROM products GROUP BY category"),
    _q("Group by", "medium", "How many orders are there for each status?",
       "SELECT status, COUNT(*) FROM orders GROUP BY status"),
    _q("Group by", "medium", "What is the average discount for each order status?",
       "SELECT status, AVG(discount) FROM orders GROUP BY status"),
    _q("Group by", "medium", "How many customers are there in each segment?",
       "SELECT segment, COUNT(*) FROM customers GROUP BY segment"),
    _q("Group by", "medium", "How many orders did each customer place? Show the customer id and the count.",
       "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id"),

    # ---------------- Join (8) ----------------
    _q("Join", "medium", "What is the total revenue for each product category?",
       "SELECT p.category, SUM(o.total_amount) FROM orders o JOIN products p ON o.product_id = p.product_id GROUP BY p.category"),
    _q("Join", "medium", "What is the total revenue for each customer city?",
       "SELECT c.city, SUM(o.total_amount) FROM orders o JOIN customers c ON o.customer_id = c.customer_id GROUP BY c.city"),
    _q("Join", "medium", "What is the total revenue for each customer segment?",
       "SELECT c.segment, SUM(o.total_amount) FROM orders o JOIN customers c ON o.customer_id = c.customer_id GROUP BY c.segment"),
    _q("Join", "medium", "How many orders were placed by customers from Mumbai?",
       "SELECT COUNT(*) FROM orders o JOIN customers c ON o.customer_id = c.customer_id WHERE c.city = 'Mumbai'"),
    _q("Join", "medium", "What is the total quantity sold in each product category?",
       "SELECT p.category, SUM(o.quantity) FROM orders o JOIN products p ON o.product_id = p.product_id GROUP BY p.category"),
    _q("Join", "hard", "For each product name, how many orders include it?",
       "SELECT p.product_name, COUNT(*) FROM orders o JOIN products p ON o.product_id = p.product_id GROUP BY p.product_name"),
    _q("Join", "medium", "What is the average order amount for Electronics products?",
       "SELECT AVG(o.total_amount) FROM orders o JOIN products p ON o.product_id = p.product_id WHERE p.category = 'Electronics'"),
    _q("Join", "hard", "How many different customers have bought Furniture products?",
       "SELECT COUNT(DISTINCT o.customer_id) FROM orders o JOIN products p ON o.product_id = p.product_id WHERE p.category = 'Furniture'"),

    # ---------------- Date & time (8) ----------------
    _q("Date & time", "medium", "How many orders were placed in 2026?",
       "SELECT COUNT(*) FROM orders WHERE EXTRACT(year FROM order_date) = 2026"),
    _q("Date & time", "medium", "What is the total revenue for each year?",
       "SELECT EXTRACT(year FROM order_date), SUM(total_amount) FROM orders GROUP BY 1"),
    _q("Date & time", "hard", "How many orders were placed in each month of 2026? Show the month number and the count.",
       "SELECT EXTRACT(month FROM order_date), COUNT(*) FROM orders WHERE EXTRACT(year FROM order_date) = 2026 GROUP BY 1"),
    _q("Date & time", "easy", "On what date was the earliest order placed?",
       "SELECT MIN(order_date) FROM orders"),
    _q("Date & time", "easy", "On what date was the most recent order placed?",
       "SELECT MAX(order_date) FROM orders"),
    _q("Date & time", "medium", "How many customers signed up in 2025?",
       "SELECT COUNT(*) FROM customers WHERE EXTRACT(year FROM signup_date) = 2025"),
    _q("Date & time", "hard", "What is the total revenue in the first quarter of 2026, from January to March?",
       "SELECT SUM(total_amount) FROM orders WHERE order_date >= DATE '2026-01-01' AND order_date < DATE '2026-04-01'"),
    _q("Date & time", "hard", "How many orders were placed on a Saturday or Sunday?",
       "SELECT COUNT(*) FROM orders WHERE dayofweek(order_date) IN (0, 6)"),

    # ---------------- Ranking (10) ----------------
    _q("Ranking", "hard", "Which 3 customer cities have the highest total revenue?",
       "SELECT c.city, SUM(o.total_amount) AS revenue FROM orders o JOIN customers c ON o.customer_id = c.customer_id GROUP BY c.city ORDER BY revenue DESC LIMIT 3",
       ordered=True),
    _q("Ranking", "hard", "Which product has the highest total revenue?",
       "SELECT p.product_name, SUM(o.total_amount) AS revenue FROM orders o JOIN products p ON o.product_id = p.product_id GROUP BY p.product_name ORDER BY revenue DESC LIMIT 1",
       ordered=True),
    _q("Ranking", "medium", "Show the 5 most expensive products with their prices.",
       "SELECT product_name, unit_price FROM products ORDER BY unit_price DESC LIMIT 5",
       ordered=True),
    _q("Ranking", "hard", "Who are the top 5 customers by total revenue? Show the customer name and the revenue.",
       "SELECT c.customer_name, SUM(o.total_amount) AS revenue FROM orders o JOIN customers c ON o.customer_id = c.customer_id GROUP BY c.customer_name ORDER BY revenue DESC LIMIT 5",
       ordered=True),
    _q("Ranking", "hard", "Which product category has the lowest total revenue?",
       "SELECT p.category, SUM(o.total_amount) AS revenue FROM orders o JOIN products p ON o.product_id = p.product_id GROUP BY p.category ORDER BY revenue ASC LIMIT 1",
       ordered=True),
    _q("Ranking", "medium", "What are the 2 biggest orders by amount? Show the order id and the amount.",
       "SELECT order_id, total_amount FROM orders ORDER BY total_amount DESC LIMIT 2",
       ordered=True),
    _q("Ranking", "hard", "Which month of 2026 had the highest revenue? Show the month number and the revenue.",
       "SELECT EXTRACT(month FROM order_date) AS m, SUM(total_amount) AS revenue FROM orders WHERE EXTRACT(year FROM order_date) = 2026 GROUP BY m ORDER BY revenue DESC LIMIT 1",
       ordered=True),
    _q("Ranking", "hard", "Which customer segment has the highest average order amount?",
       "SELECT c.segment, AVG(o.total_amount) AS avg_amount FROM orders o JOIN customers c ON o.customer_id = c.customer_id GROUP BY c.segment ORDER BY avg_amount DESC LIMIT 1",
       ordered=True),
    _q("Ranking", "medium", "List the 3 cheapest products with their prices, cheapest first.",
       "SELECT product_name, unit_price FROM products ORDER BY unit_price ASC LIMIT 3",
       ordered=True),
    _q("Ranking", "hard", "Which order status has the highest total revenue?",
       "SELECT status, SUM(total_amount) AS revenue FROM orders GROUP BY status ORDER BY revenue DESC LIMIT 1",
       ordered=True),
]

QUESTIONS = [dict(q, id=index) for index, q in enumerate(_RAW, start=1)]
