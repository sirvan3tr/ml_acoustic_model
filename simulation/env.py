import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# Speed of sound in m/s (approximate for typical conditions)
SPEED_OF_SOUND = 343.0

@dataclass
class Microphone:
    channel: int
    name: str
    x: float
    y: float
    z: float
    azimuth_deg: float # Direction it points (0 = North/Y+)
    elevation_deg: float # Direction it points up (0 = horizontal)

class VirtualMicArray:
    """
    Simulates the 8-channel microphone array used in the dataset.
    Coordinates match the relative geometry from label.json.
    """
    def __init__(self):
        # Base radius from center is approx 1.72m for all, we will use a simplified circular layout
        # matching the stated Azimuths in the JSON
        self.radius = 1.72
        
        # Lower ring (1.49m height)
        # Azimuths: 0, 90, 180, 270
        lower_height = 1.49
        
        # Upper ring (2.38m height)
        # Azimuths: 45, 135, 225, 315
        upper_height = 2.38
        
        self.mics = [
            # Channel 1-4 (Lower ring)
            self._create_mic(1, lower_height, 0.0),
            self._create_mic(2, lower_height, 90.0),
            self._create_mic(3, lower_height, 180.0),
            self._create_mic(4, lower_height, 270.0),
            # Channel 5-8 (Upper ring, elevated 20 deg per JSON)
            self._create_mic(5, upper_height, 45.0, elevation_deg=20.0),
            self._create_mic(6, upper_height, 135.0, elevation_deg=20.0),
            self._create_mic(7, upper_height, 225.0, elevation_deg=20.0),
            self._create_mic(8, upper_height, 315.0, elevation_deg=20.0),
        ]
        
    def _create_mic(self, channel: int, z: float, azimuth_deg: float, elevation_deg: float = 10.0) -> Microphone:
        # Standard polar to cartesian
        # 0 azimuth in JSON seems to align with Y axis (North) or X axis. 
        # Let's define 0 azimuth as +X, 90 as +Y
        az_rad = np.deg2rad(azimuth_deg)
        x = self.radius * np.cos(az_rad)
        y = self.radius * np.sin(az_rad)
        return Microphone(
            channel=channel, 
            name="NTG-2", 
            x=x, 
            y=y, 
            z=z, 
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg
        )

    def get_mic_positions(self) -> np.ndarray:
        """Returns (8, 3) array of [x, y, z] for each mic"""
        return np.array([[m.x, m.y, m.z] for m in self.mics])

    def get_mic_directions(self) -> np.ndarray:
        """Returns (8, 3) array of normalized direction vectors for each mic"""
        dirs = []
        for m in self.mics:
            az_rad = np.deg2rad(m.azimuth_deg)
            el_rad = np.deg2rad(m.elevation_deg)
            # spherical to cartesian direction vector
            dx = np.cos(el_rad) * np.cos(az_rad)
            dy = np.cos(el_rad) * np.sin(az_rad)
            dz = np.sin(el_rad)
            dirs.append([dx, dy, dz])
        return np.array(dirs)

def compute_propagation(source_pos: np.ndarray, array: VirtualMicArray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes time delay and amplitude attenuation for a source position.
    
    Args:
        source_pos: (3,) array [x, y, z]
        array: VirtualMicArray instance
        
    Returns:
        delays: (8,) array of time delays in seconds
        attenuations: (8,) array of linear amplitude scalings
    """
    mic_pos = array.get_mic_positions()
    mic_dirs = array.get_mic_directions()
    
    # 1. Distances to each mic
    vectors_to_source = source_pos - mic_pos # (8, 3)
    distances = np.linalg.norm(vectors_to_source, axis=1) # (8,)
    
    # 2. Delays
    delays = distances / SPEED_OF_SOUND
    
    # 3. Distance Attenuation (Inverse square law for intensity -> 1/r for amplitude)
    # Clamp minimum distance to avoid singularity
    safe_distances = np.maximum(distances, 0.1)
    dist_attenuation = 1.0 / safe_distances
    
    # 4. Angular Attenuation (Supercardioid polar pattern approximation)
    # Normalize vectors to source
    dir_to_source = vectors_to_source / safe_distances[:, np.newaxis]
    
    # Cosine of angle between mic forward direction and source direction
    cos_theta = np.sum(mic_dirs * dir_to_source, axis=1)
    
    # Typical supercardioid polar response: r(θ) = 0.37 + 0.63 * cos(θ)
    polar_response = 0.37 + 0.63 * cos_theta
    # Ensure it's non-negative and take absolute value (lobes)
    angular_attenuation = np.abs(polar_response)
    
    total_attenuation = dist_attenuation * angular_attenuation
    
    # Normalize attenuation so it's not tiny (e.g., relative to 1m distance on-axis)
    # If source is at 1m directly in front of mic, atten = 1.0
    
    return delays, total_attenuation

if __name__ == "__main__":
    array = VirtualMicArray()
    print("Mic positions:\n", array.get_mic_positions())
    
    test_source = np.array([10.0, 0.0, 1.49]) # 10m directly out on X axis
    delays, attens = compute_propagation(test_source, array)
    print("\nSource at (10, 0, 1.49) [Facing Mic 1]")
    print(f"Delays (ms): {delays * 1000}")
    print(f"Attenuations: {attens}")
