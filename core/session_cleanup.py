from core.memory_manager import get_memory_manager


def cleanup_old_sessions(days_old: int = 30) -> dict:
    """
    Clean up sessions older than specified days
    
    Args:
        days_old: Delete sessions older than this (default 30)
        
    Returns:
        Dict with cleanup stats
    """
    try:
        memory_manager = get_memory_manager()
        result = memory_manager.cleanup_old_sessions(days_old=days_old)
        return result
    except Exception as e:
        print(f"Session cleanup failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # Can be run manually
    print("Running session cleanup...")
    result = cleanup_old_sessions(days_old=30)
    
    if result['success']:
        print(f"✓ Cleanup complete: {result['sessions_deleted']} sessions deleted")
    else:
        print(f"✗ Cleanup failed: {result.get('error')}")