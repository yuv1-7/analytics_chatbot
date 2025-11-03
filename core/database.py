import os
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
from typing import Optional

# PostgreSQL connection pool
_connection_pool: Optional[psycopg2.pool.SimpleConnectionPool] = None


def initialize_connection_pool():
    """Initialize the connection pool"""
    global _connection_pool
    
    if _connection_pool is None:
        _connection_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
    
    return _connection_pool


def get_connection_pool():
    """Get or create connection pool"""
    if _connection_pool is None:
        return initialize_connection_pool()
    return _connection_pool


@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    pool = get_connection_pool()
    conn = pool.getconn()
    
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        pool.putconn(conn)


def close_connection_pool():
    """Close all connections in the pool"""
    global _connection_pool
    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None