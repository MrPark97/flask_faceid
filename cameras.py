import psycopg2
import config


def get_all_cameras():
    """get cameras' ids and urls"""

    conn = None
    try:
        params = config.db_config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()

        query = "SELECT * from cameras"
        cur.execute(query)

        rows = cur.fetchall()

        cur.close()
    except(Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

    return rows