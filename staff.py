import psycopg2
import config


def get_staff_embeddings():
    """get staff ids and embeddings"""

    conn = None
    try:
        params = config.db_config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()

        query = "SELECT staff_id, staff_embedding from staff"
        cur.execute(query)

        rows = cur.fetchall()

        cur.close()
    except(Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

    return rows


def get_staff():
    """get staff"""

    conn = None
    try:
        params = config.db_config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()

        query = "SELECT * from staff"
        cur.execute(query)

        rows = cur.fetchall()

        cur.close()
    except(Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

    return rows