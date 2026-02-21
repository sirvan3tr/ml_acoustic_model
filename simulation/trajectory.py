import numpy as np

def generate_orbit(radius: float, height: float, speed: float, duration: float, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a circular orbit around the array center.
    
    Args:
        radius: Distance from center in meters
        height: Height in meters
        speed: Speed in meters/second
        duration: Duration in seconds
        sample_rate: Control rate in Hz (e.g., 100 for 100Hz trajectory updates)
        
    Returns:
        times: (N,) array of time points
        positions: (N, 3) array of [x, y, z] coordinates
    """
    times = np.arange(0, duration, 1.0 / sample_rate)
    
    # Circumference = 2 * pi * radius
    # angular_velocity = speed / radius (rad/s)
    angular_velocity = speed / radius
    
    angles = angular_velocity * times
    
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.full_like(x, height)
    
    positions = np.stack([x, y, z], axis=1)
    return times, positions

def generate_flyby(start_pos: tuple[float, float, float], 
                   end_pos: tuple[float, float, float], 
                   duration: float, 
                   sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a straight-line flyby from start_pos to end_pos.
    
    Args:
        start_pos: (x, y, z) starting coordinate in meters
        end_pos: (x, y, z) ending coordinate in meters
        duration: Duration in seconds
        sample_rate: Control rate in Hz
        
    Returns:
        times: (N,) array of time points
        positions: (N, 3) array of [x, y, z] coordinates
    """
    times = np.arange(0, duration, 1.0 / sample_rate)
    
    # Linear interpolation
    t_norm = times / duration
    
    start_arr = np.array(start_pos)
    end_arr = np.array(end_pos)
    
    positions = start_arr[np.newaxis, :] + (end_arr - start_arr)[np.newaxis, :] * t_norm[:, np.newaxis]
    return times, positions

def generate_hover(pos: tuple[float, float, float], 
                   duration: float, 
                   sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Generates a static hovering trajectory."""
    times = np.arange(0, duration, 1.0 / sample_rate)
    positions = np.tile(np.array(pos), (len(times), 1))
    return times, positions

if __name__ == "__main__":
    t, pos = generate_orbit(15.0, 10.0, 5.0, 10.0, 100)
    print(f"Generated orbit: {len(t)} points, final pos: {pos[-1]}")
    
    t, pos = generate_flyby((-20.0, 5.0, 15.0), (20.0, -5.0, 15.0), 10.0, 100)
    print(f"Generated flyby: {len(t)} points, final pos: {pos[-1]}")
