using Godot;
using System;
using System.Collections.Generic;

public static class ParticleVolumeEditor
{
	private static readonly Random Rng = new Random();

	/// <summary>
	/// Adds particles inside a 3D Box (AABB) with optional spatial jitter to prevent lattice alignment.
	/// </summary>
	public static void AddBox(
		Vector3 center, 
		Vector3 size, 
		float spacing, 
		List<Vector4> posMass, 
		List<Vector4> velVol,
		List<uint> matIDs,
		float materialDensity,
		uint matID = 0,
		float jitterFraction = 0.35f,
		int maxParticleCap = int.MaxValue
		) // Jitter relative to spacing (e.g. 0.1 = +/-10% of spacing)
	{
		Vector3 halfSize = size * 0.5f;
		Vector3 min = center - halfSize;
		Vector3 max = center + halfSize;
		float particleVolume = spacing * spacing * spacing;
		float particleMass = materialDensity * particleVolume;
		float maxJitter = spacing * jitterFraction;

		for (float x = min.X; x <= max.X; x += spacing)
		{
			for (float y = min.Y; y <= max.Y; y += spacing)
			{
				for (float z = min.Z; z <= max.Z; z += spacing)
				{
					if (posMass.Count >= maxParticleCap) return;

					Vector3 pos = new Vector3(x, y, z) + GetRandomJitter(maxJitter);

					posMass.Add(new Vector4(pos.X, pos.Y, pos.Z, particleMass)); // Mass = 1.0
					velVol.Add(new Vector4(0f, 0f, 0f, particleVolume));
					matIDs.Add(matID);
				}
			}
		}
	}

	/// <summary>
	/// Adds particles inside a Sphere with optional spatial jitter to prevent lattice alignment.
	/// </summary>
	public static void AddSphere(
		Vector3 center, 
		float radius, 
		float spacing, 
		List<Vector4> posMass, 
		List<Vector4> velVol,
		List<uint> matIDs,
		float materialDensity,
		uint matID,
		float jitterFraction = 0.35f,
		int maxParticleCap = int.MaxValue
		)
	{
		float radiusSq = radius * radius;
		Vector3 min = center - new Vector3(radius, radius, radius);
		Vector3 max = center + new Vector3(radius, radius, radius);
		float particleVolume = spacing * spacing * spacing;
		float particleMass = materialDensity * particleVolume;
		float maxJitter = spacing * jitterFraction;

		for (float x = min.X; x <= max.X; x += spacing)
		{
			for (float y = min.Y; y <= max.Y; y += spacing)
			{
				for (float z = min.Z; z <= max.Z; z += spacing)
				{
					if (posMass.Count >= maxParticleCap) return;

					Vector3 gridPos = new Vector3(x, y, z);
					if (gridPos.DistanceSquaredTo(center) <= radiusSq)
					{
						Vector3 pos = gridPos + GetRandomJitter(maxJitter);

						posMass.Add(new Vector4(pos.X, pos.Y, pos.Z, particleMass));
						velVol.Add(new Vector4(0f, 0f, 0f, particleVolume));
						matIDs.Add(matID);
					}
				}
			}
		}
	}

	/// <summary>
	/// Removes all particles located inside a 3D Box (Carving/Digging).
	/// </summary>
	public static void RemoveBox(Vector3 center, Vector3 size, List<Vector4> posMass, List<Vector4> velVol, List<uint> matIDs)
	{
		Vector3 halfSize = size * 0.5f;
		Vector3 min = center - halfSize;
		Vector3 max = center + halfSize;

		for (int i = posMass.Count - 1; i >= 0; i--)
		{
			Vector4 p = posMass[i];
			if (p.X >= min.X && p.X <= max.X &&
				p.Y >= min.Y && p.Y <= max.Y &&
				p.Z >= min.Z && p.Z <= max.Z)
			{
				// Swap-and-pop for fast removal O(1)
				int lastIdx = posMass.Count - 1;
				posMass[i] = posMass[lastIdx];
				velVol[i] = velVol[lastIdx];
				matIDs[i] = matIDs[lastIdx];

				posMass.RemoveAt(lastIdx);
				velVol.RemoveAt(lastIdx);
				matIDs.RemoveAt(lastIdx);
			}
		}
	}

	/// <summary>
	/// Removes all particles located inside a Sphere (Crater/Explosion).
	/// </summary>
	public static void RemoveSphere(Vector3 center, float radius, List<Vector4> posMass, List<Vector4> velVol, List<uint> matIDs)
	{
		float radiusSq = radius * radius;

		for (int i = posMass.Count - 1; i >= 0; i--)
		{
			Vector4 p = posMass[i];
			Vector3 pos = new Vector3(p.X, p.Y, p.Z);

			if (pos.DistanceSquaredTo(center) <= radiusSq)
			{
				// Swap-and-pop for fast removal O(1)
				int lastIdx = posMass.Count - 1;
				posMass[i] = posMass[lastIdx];
				velVol[i] = velVol[lastIdx];
				matIDs[i] = matIDs[lastIdx];

				posMass.RemoveAt(lastIdx);
				velVol.RemoveAt(lastIdx);
				matIDs.RemoveAt(lastIdx);
			}
		}
	}

	/// <summary>
	/// Generates a random 3D perturbation vector within [-maxJitter, +maxJitter].
	/// </summary>
	private static Vector3 GetRandomJitter(float maxJitter)
	{
		if (maxJitter <= 0f) return Vector3.Zero;

		float rx = (float)(Rng.NextDouble() * 2.0 - 1.0) * maxJitter;
		float ry = (float)(Rng.NextDouble() * 2.0 - 1.0) * maxJitter;
		float rz = (float)(Rng.NextDouble() * 2.0 - 1.0) * maxJitter;

		return new Vector3(rx, ry, rz);
	}
}
