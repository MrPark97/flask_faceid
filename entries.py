import psycopg2
import config
import staff
import datetime

def dot(x, y):
    """Dot product as sum of list comprehension doing element-wise multiplication"""
    return sum(x_i * y_i for x_i, y_i in zip(x, y))


def get_last_entries():
    """compare last 100 entries with database"""

    conn = None
    try:
        params = config.db_config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()

        query = "SELECT * from entries WHERE DATE(entry_date) >= CURRENT_DATE"
        cur.execute(query)

        entries = cur.fetchall()

        cur.close()
    except(Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

    recognized_entries = []

    # load staff embeddings from db
    staff_rows = staff.get_staff()
    for entry in entries:
        max_sim = 0.0
        sim_index = 1000000000
        sim_name = ""
        sim_image = ""
        for staff_row in staff_rows:
            cos_sim = dot(entry[1], staff_row[3])
            if cos_sim > max_sim:
                max_sim = cos_sim
                sim_index = staff_row[0]
                sim_name = staff_row[1]
                sim_image = staff_row[2]

        if max_sim >= 0.45:
            recognized_entry = {'confidence' : max_sim, 'name': sim_name, 'datetime': entry[2], 'image': sim_image}
            recognized_entries.append(recognized_entry)

    return recognized_entries
