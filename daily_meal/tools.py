# tools.py
import sqlite3
import datetime
import json

# ==============================================================================
# 数据库操作辅助函数 (与之前基本相同，进行了一些封装和打印优化)
# ==============================================================================

DB_NAME = "food_storage_and_meals.db"  # 统一数据库文件名


def _connect_db():
    """内部函数：连接到数据库并返回连接和游标。"""
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row  # 使查询结果可以通过字典方式访问
        cursor = conn.cursor()
        return conn, cursor
    except sqlite3.Error as e:
        print(f"数据库连接错误: {e}")
        raise


def _execute_query(query, params=(), commit=False):
    """内部函数：执行SQL查询，处理连接和关闭。"""
    conn, cursor = _connect_db()
    try:
        cursor.execute(query, params)
        if commit:
            conn.commit()
            return cursor.lastrowid  # 返回最后插入的ID
        else:
            rows = cursor.fetchall()
            return [dict(row) for row in rows]  # 返回字典列表
    except sqlite3.Error as e:
        print(f"数据库操作错误: {e}")
        conn.rollback()  # 回滚事务
        return None
    finally:
        conn.close()


def _create_tables():
    """创建foods和meal_records表（如果不存在）。"""
    conn, cursor = _connect_db()
    try:
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS foods
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           user_id
                           INTEGER
                           NOT
                           NULL,
                           name
                           TEXT
                           NOT
                           NULL,
                           description
                           TEXT,
                           create_time
                           TEXT
                           NOT
                           NULL
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           update_time
                           TEXT
                           NOT
                           NULL
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           status
                           INTEGER
                           NOT
                           NULL
                           DEFAULT
                           1
                       )
                       ''')
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS meal_records
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           user_id
                           INTEGER
                           NOT
                           NULL,
                           meal_type
                           TEXT
                           NOT
                           NULL,
                           food_id
                           INTEGER
                           NOT
                           NULL,
                           quantity
                           REAL
                           NOT
                           NULL,
                           status
                           INTEGER
                           NOT
                           NULL
                           DEFAULT
                           1,
                           create_time
                           TEXT
                           NOT
                           NULL
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           update_time
                           TEXT
                           NOT
                           NULL
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           FOREIGN
                           KEY
                       (
                           food_id
                       ) REFERENCES foods
                       (
                           id
                       ) ON DELETE RESTRICT
                           )
                       ''')
        conn.commit()
        print("数据库表 'foods' 和 'meal_records' 已准备就绪。")
    except sqlite3.Error as e:
        print(f"创建数据库表错误: {e}")
        raise
    finally:
        conn.close()


# 在模块加载时确保表存在
_create_tables()


# ==============================================================================
# Food Storage Tools
# ==============================================================================

def add_food_tool(user_id: int, name: str, description: str = "") -> dict:
    """
    添加一个新的食品记录。
    :param user_id: 用户ID。
    :param name: 食品名称。
    :param description: 食品描述（可选）。
    :return: 包含操作结果的字典。
    """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    query = '''
            INSERT INTO foods (user_id, name, description, create_time, update_time, status)
            VALUES (?, ?, ?, ?, ?, ?) \
            '''
    food_id = _execute_query(query, (user_id, name, description, current_time, current_time, 1), commit=True)
    if food_id is not None:
        return {"status": "success", "message": f"成功添加食品: ID={food_id}, 名称='{name}'", "food_id": food_id}
    else:
        return {"status": "error", "message": f"添加食品 '{name}' 失败。"}


def get_food_by_id_tool(food_id: int) -> dict:
    """
    根据食品ID获取单个食品记录。
    :param food_id: 食品ID。
    :return: 包含食品数据的字典，如果未找到则status为error。
    """
    query = "SELECT * FROM foods WHERE id = ? AND status = 1"
    result = _execute_query(query, (food_id,))
    if result:
        return {"status": "success", "data": result[0]}
    else:
        return {"status": "error", "message": f"未找到ID为 {food_id} 的食品记录。"}


def get_food_by_name_tool(user_id: int, name: str) -> dict:
    """
    根据食品名称获取用户的食品记录。
    :param user_id: 用户ID。
    :param name: 食品名称。
    :return: 包含食品数据的字典列表，如果未找到则status为error。
    """
    query = "SELECT * FROM foods WHERE user_id = ? AND name LIKE ? AND status = 1"
    # 使用LIKE进行模糊匹配，更灵活
    result = _execute_query(query, (user_id, f"%{name}%"))
    if result:
        return {"status": "success", "data": result}
    else:
        return {"status": "error", "message": f"未找到用户 {user_id} 的食品 '{name}'。"}


# 可以添加 update_food 和 delete_food_tool，但当前需求主要聚焦于添加和查询
# def update_food_tool(...)
# def delete_food_tool(...)

# ==============================================================================
# Meal Tracking Tools
# ==============================================================================

def add_meal_record_tool(user_id: int, meal_type: str, food_id: int, quantity: float) -> dict:
    """
    添加一个新的餐食记录。
    :param user_id: 用户ID。
    :param meal_type: 用餐类型 (如 '早餐', '午餐', '晚餐', '加餐')。
    :param food_id: 关联的食品ID。
    :param quantity: 食用数量。
    :return: 包含操作结果的字典。
    """
    # 在工具层再次验证food_id是否存在并活跃，确保数据完整性
    food_check = get_food_by_id_tool(food_id)
    if food_check["status"] == "error":
        return {"status": "error", "message": f"食品ID {food_id} 不存在或已被删除，无法添加餐食记录。"}

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    query = '''
            INSERT INTO meal_records (user_id, meal_type, food_id, quantity, status, create_time, update_time)
            VALUES (?, ?, ?, ?, ?, ?, ?) \
            '''
    record_id = _execute_query(query, (user_id, meal_type, food_id, quantity, 1, current_time, current_time),
                               commit=True)
    if record_id is not None:
        return {"status": "success", "message": f"成功添加餐食记录: ID={record_id}", "record_id": record_id}
    else:
        return {"status": "error", "message": "添加餐食记录失败。"}


def get_meal_records_tool(user_id: int, meal_type: str = None, start_date: str = None, end_date: str = None,
                          include_deleted: bool = False) -> dict:
    """
    获取用户的餐食记录，可关联食品名称。
    :param user_id: 用户ID。
    :param meal_type: 可选的餐食类型过滤。
    :param start_date: 开始日期 (YYYY-MM-DD)。
    :param end_date: 结束日期 (YYYY-MM-DD)。
    :param include_deleted: 是否包含已删除的记录 (status=0)。
    :return: 包含餐食记录列表的字典。
    """
    query = """
            SELECT mr.*, f.name AS food_name, f.description AS food_description
            FROM meal_records mr
                     JOIN foods f ON mr.food_id = f.id
            WHERE mr.user_id = ? \
            """
    params = [user_id]

    if not include_deleted:
        query += " AND mr.status = 1"

    if meal_type:
        query += " AND mr.meal_type = ?"
        params.append(meal_type)

    if start_date:
        query += " AND mr.create_time >= ?"
        params.append(start_date + " 00:00:00")

    if end_date:
        query += " AND mr.create_time <= ?"
        params.append(end_date + " 23:59:59")

    query += " ORDER BY mr.create_time DESC"

    records = _execute_query(query, params)
    if records is not None:
        return {"status": "success", "data": records}
    else:
        return {"status": "error", "message": "获取餐食记录失败。"}


def update_meal_record_tool(record_id: int, user_id: int, meal_type: str = None, food_id: int = None,
                            quantity: float = None) -> dict:
    """
    更新餐食记录。
    :param record_id: 要更新的餐食记录ID。
    :param user_id: 用户ID (用于验证)。
    :param meal_type: 新的用餐类型（可选）。
    :param food_id: 新的食品ID（可选）。
    :param quantity: 新的食用数量（可选）。
    :return: 包含操作结果的字典。
    """
    updates = []
    params = []
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if meal_type is not None:
        updates.append("meal_type = ?")
        params.append(meal_type)
    if quantity is not None:
        updates.append("quantity = ?")
        params.append(quantity)
    if food_id is not None:
        food_check = get_food_by_id_tool(food_id)
        if food_check["status"] == "error":
            return {"status": "error", "message": f"新的食品ID {food_id} 不存在或已被删除，无法更新餐食记录。"}
        updates.append("food_id = ?")
        params.append(food_id)

    if not updates:
        return {"status": "error", "message": f"没有为ID为 {record_id} 的餐食记录提供更新数据。"}

    updates.append("update_time = ?")
    params.append(current_time)

    query = f"UPDATE meal_records SET {', '.join(updates)} WHERE id = ? AND user_id = ? AND status = 1"
    params.extend([record_id, user_id])

    conn, cursor = _connect_db()
    try:
        cursor.execute(query, params)
        conn.commit()
        if cursor.rowcount > 0:
            return {"status": "success", "message": f"成功更新餐食记录: ID={record_id}"}
        else:
            return {"status": "error", "message": f"未找到ID为 {record_id} 且状态为正常的餐食记录进行更新或无权限。"}
    except sqlite3.Error as e:
        print(f"更新餐食记录错误: {e}")
        conn.rollback()
        return {"status": "error", "message": f"更新餐食记录失败: {e}"}
    finally:
        conn.close()


def delete_meal_record_tool(record_id: int, user_id: int) -> dict:
    """
    “删除”餐食记录（将状态设置为0，进行软删除）。
    :param record_id: 要删除的餐食记录ID。
    :param user_id: 用户ID (用于验证)。
    :return: 包含操作结果的字典。
    """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    query = "UPDATE meal_records SET status = 0, update_time = ? WHERE id = ? AND user_id = ? AND status = 1"

    conn, cursor = _connect_db()
    try:
        cursor.execute(query, (current_time, record_id, user_id))
        conn.commit()
        if cursor.rowcount > 0:
            return {"status": "success", "message": f"成功“删除”餐食记录 (软删除): ID={record_id}"}
        else:
            return {"status": "error", "message": f"未找到ID为 {record_id} 且状态为正常的餐食记录进行删除或无权限。"}
    except sqlite3.Error as e:
        print(f"删除餐食记录错误: {e}")
        conn.rollback()
        return {"status": "error", "message": f"删除餐食记录失败: {e}"}
    finally:
        conn.close()


def get_meal_record_by_id_tool(record_id: int, user_id: int) -> dict:
    """
    根据记录ID获取单个餐食记录。
    :param record_id: 餐食记录ID。
    :param user_id: 用户ID (用于验证)。
    :return: 餐食记录字典，如果未找到则返回None。
    """
    query = """
            SELECT mr.*, f.name AS food_name, f.description AS food_description
            FROM meal_records mr
                     JOIN foods f ON mr.food_id = f.id
            WHERE mr.id = ? \
              AND mr.user_id = ? \
              AND mr.status = 1 \
            """
    result = _execute_query(query, (record_id, user_id))
    if result:
        return {"status": "success", "data": result[0]}
    else:
        return {"status": "error", "message": f"未找到ID为 {record_id} 的餐食记录或无权限。"}