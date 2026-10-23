using Godot;
using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;

public partial class MpmSolver : Node3D
{
	[StructLayout(LayoutKind.Explicit, Size = 80)]
	public struct SimulationParams
	{
		// Block 1 (0..15)
		[FieldOffset(0)]  public int GridDimX;
		[FieldOffset(4)]  public int GridDimY;
		[FieldOffset(8)]  public int GridDimZ;
		[FieldOffset(12)] public float Dx;

		// Block 2 (16..31)
		[FieldOffset(16)] public float WorldOffsetX;
		[FieldOffset(20)] public float WorldOffsetY;
		[FieldOffset(24)] public float WorldOffsetZ;
		[FieldOffset(28)] public float Dt;

		// Block 3 (32..47)
		[FieldOffset(32)] public uint TotalParticles;
		[FieldOffset(36)] public float InvDx;
		[FieldOffset(40)] public float Padding1;
		[FieldOffset(44)] public float Padding2;
		[FieldOffset(48)] public float ImpulsePositionX;
		[FieldOffset(52)] public float ImpulsePositionY;
		[FieldOffset(56)] public float ImpulsePositionZ;
		[FieldOffset(60)] public float ImpulseRadius;
		[FieldOffset(64)] public float ImpulseDirectionX;
		[FieldOffset(68)] public float ImpulseDirectionY;
		[FieldOffset(72)] public float ImpulseDirectionZ;
		[FieldOffset(76)] public float ImpulseStrength;
	}

	public struct MaterialProps {
		public float E;           // Young's Modulus
		public float NU;          // Poisson's Ratio
		public float friction;    // Friction angle (for sand)
		public float cohesion;    // Cohesion strength
		public float density;     // Density in kg/m^3
		public uint model_type;   // 0 = Elastic, 1 = Drucker-Prager, 2 = Fluid
	};
	
	[ExportGroup("Simulation Settings")]

	[Export] public float DomainSize = 4.0f;
	[Export] public int GridDim = 64;
	[Export] public int ParticlesPerCellAxis = 2; 

	[Export] public int MaxParticles = 1_000_000;

	[Export] public float MaxParticleSpeed = 500.0f;
	[Export] public float SimulationSpeed = 1.0f;


	[Export] public int MaxSubstepsPerFrame = 1;
	[ExportGroup("Particle Impulse")]
	[Export] public float ImpulseAimDistance = 0.8f;
	[Export] public float ImpulseRadius = 0.48f;
	[Export] public float ImpulseStrength = 5.0f;

	[ExportGroup("Position Based MPM")]
	[Export] public int PBIterations = 8;

	[ExportGroup("Rendering Settings")]
	[Export] public float MeshUpdateHz = 30.0f; // Rate-limit GPU-to-CPU mesh extraction

	[ExportGroup("GPU Mesh")]
	[Export] public int RenderVertexCapacity = 4_000_008;
	// Width of the linearized vertex-data textures.
	// Height is calculated automatically.
	[Export] public uint MeshDataTextureWidth = 2048;

	private float _meshUpdateTimer = 0.0f;
	private bool _impulsePending;
	private Vector3 _impulsePosition;
	private Vector3 _impulseDirection;

	private MaterialProps[] physMats = new []{
		// Index 0: Soft Rubber / Jelly (Elastic Solid)
		new MaterialProps {
			E = 1e4f,          // Young's Modulus: Moderate stiffness
			NU = 0.3f,         // Poisson's Ratio: Moderate lateral expansion
			friction = 0.0f,   // Unused for Elastic
			cohesion = 0.0f,
			density = 900.0f,
			model_type = 0     // 0 = Elastic Solid
		},

		// Index 1: Stiff Plastic / Wood (Rigid Elastic Solid)
		new MaterialProps {
			E = 6e5f,          // Young's Modulus: High stiffness (resists warping)
			NU = 0.25f,        // Poisson's Ratio: Lower lateral expansion
			friction = 0.0f,   // Unused for Elastic
			cohesion = 0.0f,
			density = 700.0f,
			model_type = 0     // 0 = Elastic Solid
		},

		// Index 2: Dry Sand (Granular)
		new MaterialProps {
			E = 1e4f,          // Stiffness when compressed
			NU = 0.3f,         // Poisson's Ratio
			friction = 35.0f,  // Friction angle in degrees (sets natural angle of repose ~35°)
			cohesion = 0.0f,
			density = 1600.0f,
			model_type = 1     // 1 = Drucker-Prager Sand
		},

		// Index 3: Cohesive Clay (Granular Plasticity with Cohesion)
		new MaterialProps {
			E = 1.5e7f,          // Moderate-high stiffness (holds shape well under small loads)
			NU = 0.38f,        // Higher Poisson's ratio (incompressible/rubbery resistance when squished)
			friction = 30.0f,  // Very high friction angle (simulates strong shear resistance)
			cohesion = 1.5e6f,
			density = 1800.0f,
			model_type = 1     // 1 = Drucker-Prager Plasticity
		},

		// Index 4: Water (Fluid)
		new MaterialProps {
			E = 5.0e5f,        // 5 MPa (Yields c_s ~ 70m/s -> Keeps fluid incompressible under gravity)
			NU = 0.001f,       // Low artificial viscosity (0.001 = Thin, splashing liquid)
			friction = 0.0f,   // Unused for Fluid
			cohesion = 0.0f,   // Unused for Fluid (or tiny value like 10.0 for surface tension)
			density = 1000.0f, // 1000 kg/m^3 (Standard water)
			model_type = 2     // 2 = Weakly Compressible Fluid (Tait EOS)
		},

		// Index 5: Thick Mud / Honey (Viscous Fluid)
		new MaterialProps {
			E = 2e4f,          // Higher bulk stiffness
			NU = 0.4f,         // High viscosity (flows slowly like honey or wet mud)
			friction = 0.0f,   // Unused for Fluid
			cohesion = 0.0f,
			density = 1800.0f,
			model_type = 2     // 2 = Weakly Compressible Fluid
		}
	};

	private float stableDt = 1.0f / 90.0f;
	private float _substepDt;

	// Helper properties
	public int TotalGridCells => GridDim * GridDim * GridDim;
	public uint MaxMcVertices => (uint)(TotalGridCells * 15); // Max potential MC vertices

	// State Tracking
	private int _activeParticleCount = 0;


	// Godot RenderingDevice Handle
	private RenderingDevice _rd;

	// Storage Buffers
	private Rid _posMassBuffer;
	private Rid _velVolBuffer;
	private Rid _affineBuffer;
	private Rid _elasticFBuffer;
	private Rid _stressBuffer;
	private Rid _gridBuffer;
	private Rid _solvedGridBuffer;
	private Rid _densityBuffer;
	private Rid _triTableBuffer;
	private Rid _physMatsBuffer;
	private Rid _matIDsBuffer;
	private Rid _paramsBuffer;
	private Rid _deformationBuffer;

	// Compute Pipelines
	private Rid _p2gPipeline;
	private Rid _gridSolvePipeline;
	private Rid _g2pPipeline;
	private Rid _densityPipeline;
	private Rid _marchingCubesPipeline;
	private Rid _constraintPipeline;

	// Uniform Sets
	private Rid _p2gUniformSet;
	private Rid _gridSolveUniformSet;
	private Rid _g2pUniformSet;
	private Rid _densityUniformSet;
	private Rid _constraintUniformSet;

	// Shaders
	private Rid _shaderP2G;
	private Rid _shaderGrid;
	private Rid _shaderG2P;
	private Rid _shaderDensity;
	private Rid _shaderMC;
	private Rid _shaderConstraint;

	// ----------
	private uint MeshDataTextureHeight =>
		(uint) Mathf.CeilToInt((float)RenderVertexCapacity / MeshDataTextureWidth);

	// GPU textures holding the generated mesh.
	private Rid _mpmParticlePositionTextureRid;
	private Rid _mpmParticleVelocityTextureRid;
	private Rid _mcPositionTextureRid;
	private Rid _mcNormalTextureRid;
	private Rid _mcVertexCountTextureRid;

	// Godot wrappers around those RenderingDevice textures.
	private Texture2Drd _mpmParticlePositionTexture;
	private Texture2Drd _mpmParticleVelocityTexture;
	private Texture2Drd _mcPositionTexture;
	private Texture2Drd _mcNormalTexture;
	private Texture2Drd _mcVertexCountTexture;

	// GPU-only mesh counter remains a normal storage buffer.
	private Rid _mcCounterBuffer;

	// Finalize-counter compute shader/pipeline.
	private Rid _shaderMCFinalize;
	private Rid _mcFinalizePipeline;

	// Uniform sets.
	private Rid _marchingCubesUniformSet;
	private Rid _mcFinalizeUniformSet;

	// Ordinary Godot mesh used as a fixed-size carrier.
	private MeshInstance3D _meshInstance;
	private MeshInstance3D _meshInstance2;
	private ArrayMesh _arrayMesh;
	private ArrayMesh _arrayMesh2;
	private ShaderMaterial _gpuMeshMaterial;
	private ShaderMaterial _gpuMeshMaterial2;
	

	// Material shader.
	private Shader _gpuMeshShader;
	private Shader _gpuMeshShader2;

	// Cached simulation bounds for Godot culling.
	private Aabb _meshAabb;
	// --------------


	public override void _Ready()
	{
		//stableDt = DeriveStableDt(physMats, DomainSize / (float)GridDim, MaxParticleSpeed);
		//stableDt = 1.0f / 90.0f;
		//stableDt /= 8.0f;
		GD.Print("StableDt set: " + (stableDt*1000) + "ms");
		_rd = RenderingServer.GetRenderingDevice();

		// Create the carrier MeshInstance3D first.
		SetupRenderMesh();
		// Create GPU buffers/textures.
		InitializeBuffers();
		// Compile compute shaders.
		LoadShaders();
		// Create compute uniform sets.
		BuildUniformSets();
	}

	private void SetupRenderMesh()
	{
		_meshInstance = new MeshInstance3D();
		_arrayMesh = new ArrayMesh();
		_meshInstance.Mesh = _arrayMesh;

		_meshInstance2 = new MeshInstance3D();
		_arrayMesh2 = new ArrayMesh();
		_meshInstance2.Mesh = _arrayMesh2;

		var carrierVertices = new Vector3[RenderVertexCapacity];
		var surfaceArrays = new Godot.Collections.Array();
		surfaceArrays.Resize((int)Mesh.ArrayType.Max);
		surfaceArrays[(int)Mesh.ArrayType.Vertex] = carrierVertices;
		_arrayMesh.AddSurfaceFromArrays(Mesh.PrimitiveType.Triangles, surfaceArrays);

		var arrays = new Godot.Collections.Array();
		Vector3[] dummyPoints = new Vector3[MaxParticles];
		arrays.Resize((int)Mesh.ArrayType.Max);
		arrays[(int)Mesh.ArrayType.Vertex] = dummyPoints;
		_arrayMesh2.AddSurfaceFromArrays(Mesh.PrimitiveType.Points, arrays);

		// -------------------------------------------------------------------------
		// GPU-backed material.
		// -------------------------------------------------------------------------
		SetupShaderMaterial(ref _meshInstance, ref _gpuMeshShader, ref _gpuMeshMaterial, "res://shaders/mpm_surface.gdshader");
		SetupShaderMaterial(ref _meshInstance2, ref _gpuMeshShader2, ref _gpuMeshMaterial2, "res://shaders/mpm_splat.gdshader");

		// -------------------------------------------------------------------------
		// Important: the carrier's actual vertex positions are all zero, so Godot's
		// normal AABB would be useless. Tell the GeometryInstance3D where the
		// generated geometry can actually appear.
		// -------------------------------------------------------------------------

		_meshAabb = new Aabb(
			new Vector3(-32.0f, -32.0f, -32.0f),
			new Vector3(64.0f, 64.0f, 64.0f)
		);

		_meshInstance.CustomAabb = _meshAabb;
		_meshInstance2.CustomAabb = _meshAabb;

		AddChild(_meshInstance);
		AddChild(_meshInstance2);
	}

	private void InitializeMeshTextures()
	{
		uint width = MeshDataTextureWidth;
		uint height = MeshDataTextureHeight;

		_mpmParticlePositionTextureRid = CreateMeshTexture(
			RenderingDevice.DataFormat.R32G32B32A32Sfloat,
			2048,
			(uint)Math.Ceiling((double)MaxParticles / 2048.0)
		);
		_mpmParticleVelocityTextureRid = CreateMeshTexture(
			RenderingDevice.DataFormat.R32G32B32A32Sfloat,
			2048,
			(uint)Math.Ceiling((double)MaxParticles / 2048.0)
		);

		_mcPositionTextureRid = CreateMeshTexture(RenderingDevice.DataFormat.R32G32B32A32Sfloat, width, height);
		_mcNormalTextureRid = CreateMeshTexture(RenderingDevice.DataFormat.R16G16B16A16Sfloat, width, height);

		// Single float texel containing the number of valid vertices.
		_mcVertexCountTextureRid = CreateMeshTexture(RenderingDevice.DataFormat.R32Sfloat, 1, 1);

		_mpmParticlePositionTexture = new Texture2Drd();
		_mpmParticlePositionTexture.TextureRdRid = _mpmParticlePositionTextureRid;		
		_mpmParticleVelocityTexture = new Texture2Drd();
		_mpmParticleVelocityTexture.TextureRdRid = _mpmParticleVelocityTextureRid;

		_mcPositionTexture = new Texture2Drd();
		_mcPositionTexture.TextureRdRid = _mcPositionTextureRid;

		_mcNormalTexture = new Texture2Drd();
		_mcNormalTexture.TextureRdRid = _mcNormalTextureRid;

		_mcVertexCountTexture = new Texture2Drd();
		_mcVertexCountTexture.TextureRdRid = _mcVertexCountTextureRid;

		// Hand the GPU textures to the ordinary Godot material.
		_gpuMeshMaterial.SetShaderParameter("mpm_position_data", _mcPositionTexture);
		_gpuMeshMaterial.SetShaderParameter("mpm_normal_data", _mcNormalTexture);
		_gpuMeshMaterial.SetShaderParameter("mpm_vertex_count",_mcVertexCountTexture);
		_gpuMeshMaterial2.SetShaderParameter("mpm_particle_positions",_mpmParticlePositionTexture);
		_gpuMeshMaterial2.SetShaderParameter("mpm_particle_velocities", _mpmParticleVelocityTexture);
		_gpuMeshMaterial2.SetShaderParameter("particle_count", _activeParticleCount);
	}

	private void InitializeBuffers()
	{
		var posMass = new List<Vector4>();
		var velVol = new List<Vector4>();
		var matIDs = new List<uint>();
		
		Vector3I GridDimensions = new Vector3I(GridDim, GridDim, GridDim);
		float MPM_DX = DomainSize / (float)GridDim;
		
		float ParticleSpacing = MPM_DX / ParticlesPerCellAxis; // 0.03125m
		float InitialParticleVolume = ParticleSpacing * ParticleSpacing * ParticleSpacing;

		float spacing = ParticleSpacing;

		
		// Add plastic Box
		ParticleVolumeEditor.AddBox(
			center: new Vector3(2f, 0.9f, 2f),
			size: new Vector3(2f, 1.0f, 2f),
			spacing: spacing,
			posMass,
			velVol,
			matIDs,
			physMats[4].density,
			4
		);
		
		// Add box of water
		ParticleVolumeEditor.AddBox(
			center: new Vector3(2f, 2.0f, 2f),
			size: new Vector3(1f, 1.0f, 1f),
			spacing: spacing,
			posMass,
			velVol,
			matIDs,
			physMats[1].density,
			1
		);

	
		/*// Add water sphere
		ParticleVolumeEditor.AddSphere(
			center: new Vector3(8.5f, 6f, 8.5f),
			radius: 2.5f,
			spacing: spacing,
			posMass,
			velVol,
			matIDs,
			physMats[4].density,
			4
		); */
		/*
		// Add Sand Sphere
		ParticleVolumeEditor.AddSphere(
			center: new Vector3(3.5f, 5f, 5.25f),
			radius: 1f,
			spacing: spacing,
			posMass,
			velVol,
			matIDs,
			physMats[2].density,
			2
		);

		// Add Clay Sphere
		ParticleVolumeEditor.AddSphere(
			center: new Vector3(6.5f, 5f, 2.5f),
			radius: 2f,
			spacing: spacing,
			posMass,
			velVol,
			matIDs,
			physMats[3].density,
			3
		);*/

		_activeParticleCount = posMass.Count;
		GD.Print("Particle Count: " + _activeParticleCount);

		var identityMatrices = new Projection[_activeParticleCount];
		Array.Fill(identityMatrices, Projection.Identity);

		byte[] identityBytes = MemoryMarshal.AsBytes(identityMatrices.AsSpan()).ToArray();
		byte[] zeroBytes = new byte[_activeParticleCount * 16 * sizeof(float)];
		byte[] posMassBytes = ToByteArray(posMass.ToArray());
		byte[] velVolBytes = ToByteArray(velVol.ToArray());
		byte[] matIDsBytes = ToByteArray(matIDs.ToArray());
		byte[] physMatsBytes = ToByteArray(physMats);


		_posMassBuffer    = CreateStorageBuffer((uint)posMassBytes.Length, posMassBytes);
		_velVolBuffer     = CreateStorageBuffer((uint)velVolBytes.Length, velVolBytes);
		_affineBuffer     = CreateStorageBuffer((uint)zeroBytes.Length, zeroBytes);
		_elasticFBuffer   = CreateStorageBuffer((uint)identityBytes.Length, identityBytes);
		_stressBuffer     = CreateStorageBuffer((uint)zeroBytes.Length, zeroBytes);

		_deformationBuffer = CreateStorageBuffer((uint)identityBytes.Length, identityBytes);

		_matIDsBuffer     = CreateStorageBuffer((uint)matIDsBytes.Length, matIDsBytes);
		_physMatsBuffer   = CreateStorageBuffer((uint)physMatsBytes.Length, physMatsBytes);

		_gridBuffer       = CreateStorageBuffer((uint)(TotalGridCells * 16 * sizeof(float)));
		_solvedGridBuffer = CreateStorageBuffer((uint)(TotalGridCells * 16 * sizeof(float)));
		_densityBuffer    = CreateStorageBuffer((uint)(TotalGridCells * sizeof(int)));


		SimulationParams simParams = new SimulationParams {
			GridDimX = GridDim,
			GridDimY = GridDim,
			GridDimZ = GridDim,
			Dx = DomainSize / (float)GridDim,
			WorldOffsetX = 0.0f,
			WorldOffsetY = 0.0f,
			WorldOffsetZ = 0.0f,
			Dt = stableDt,
			TotalParticles = (uint)_activeParticleCount,
			InvDx = 1.0f / (DomainSize / (float)GridDim)
		};

		byte[] paramBytes = StructureToByteArray(simParams);
		_paramsBuffer = CreateUniformBuffer((uint)paramBytes.Length, paramBytes);

		byte[] triBytes = ToByteArray(MarchingCubesTables.TriTable);
		_triTableBuffer  = CreateStorageBuffer((uint)triBytes.Length, triBytes);
		_mcCounterBuffer = CreateStorageBuffer(sizeof(uint));

		// GPU mesh representation.
		InitializeMeshTextures();
	}

	private void LoadShaders()
	{
		_shaderP2G        = MpmHelper.CompileShader(_rd, "res://shaders/mpm_p2g.glsl");
		_shaderGrid       = MpmHelper.CompileShader(_rd, "res://shaders/mpm_grid.glsl");
		_shaderG2P        = MpmHelper.CompileShader(_rd, "res://shaders/mpm_g2p.glsl");
		_shaderDensity    = MpmHelper.CompileShader(_rd, "res://shaders/mpm_density.glsl");
		_shaderMC         = MpmHelper.CompileShader(_rd, "res://shaders/mpm_marching_cubes.glsl");
		_shaderMCFinalize = MpmHelper.CompileShader(_rd, "res://shaders/mpm_mc_finalize.glsl");
		_shaderConstraint = MpmHelper.CompileShader(_rd, "res://shaders/mpm_constraint.glsl");

		_p2gPipeline           = _rd.ComputePipelineCreate(_shaderP2G);
		_gridSolvePipeline     = _rd.ComputePipelineCreate(_shaderGrid);
		_g2pPipeline           = _rd.ComputePipelineCreate(_shaderG2P);
		_densityPipeline       = _rd.ComputePipelineCreate(_shaderDensity);
		_marchingCubesPipeline = _rd.ComputePipelineCreate(_shaderMC);
		_mcFinalizePipeline    = _rd.ComputePipelineCreate(_shaderMCFinalize);
		_constraintPipeline    = _rd.ComputePipelineCreate(_shaderConstraint);
	}

	public void UpdateSimulationParams(
		int gridDim,
		float dx,
		float dt,
		Vector3 worldOffset,
		uint particleCount,
		float iterationFlags = 0.0f,
		Vector3 impulsePosition = default,
		Vector3 impulseDirection = default,
		float impulseRadius = 0.0f,
		float impulseStrength = 0.0f)
	{
		SimulationParams paramsData = new SimulationParams
		{
			GridDimX = gridDim, GridDimY = gridDim, GridDimZ = gridDim, Dx = dx,
			WorldOffsetX = worldOffset.X, WorldOffsetY = worldOffset.Y, WorldOffsetZ = worldOffset.Z, Dt = dt,
			TotalParticles = particleCount,
			InvDx = 1.0f / dx,
			Padding1 = iterationFlags,
			ImpulsePositionX = impulsePosition.X,
			ImpulsePositionY = impulsePosition.Y,
			ImpulsePositionZ = impulsePosition.Z,
			ImpulseRadius = impulseRadius,
			ImpulseDirectionX = impulseDirection.X,
			ImpulseDirectionY = impulseDirection.Y,
			ImpulseDirectionZ = impulseDirection.Z,
			ImpulseStrength = impulseStrength
		};

		// Convert struct to byte array
		byte[] bytes = StructureToByteArray(paramsData);

		// Upload new data directly to GPU buffer
		_rd.BufferUpdate(_paramsBuffer, 0, (uint)bytes.Length, bytes);
	}

	private void BuildUniformSets() {
		(Rid,RenderingDevice.UniformType,int) UB(Rid buffer, int binding) => (buffer, RenderingDevice.UniformType.UniformBuffer, binding);
		(Rid,RenderingDevice.UniformType,int) Img(Rid image, int binding) => (image, RenderingDevice.UniformType.Image, binding);
		(Rid,RenderingDevice.UniformType,int) SB(Rid buffer, int binding) => (buffer, RenderingDevice.UniformType.StorageBuffer, binding);

		_p2gUniformSet = CreateUniformSet(_shaderP2G, new[]{
			UB(_paramsBuffer, 0), SB(_posMassBuffer, 1), SB(_velVolBuffer, 2),
			SB(_affineBuffer, 3), SB(_stressBuffer, 4), SB(_gridBuffer, 5), SB(_deformationBuffer, 6)
		});

		_gridSolveUniformSet = CreateUniformSet(_shaderGrid, new[]{
			UB(_paramsBuffer, 0), SB(_gridBuffer, 1), SB(_solvedGridBuffer, 2)
		});

		_g2pUniformSet = CreateUniformSet(_shaderG2P, new[]{
			UB(_paramsBuffer, 0),
			SB(_posMassBuffer, 1), SB(_velVolBuffer, 2),
			SB(_affineBuffer, 3), SB(_deformationBuffer, 4), SB(_elasticFBuffer, 5), SB(_stressBuffer, 6),
			SB(_solvedGridBuffer, 7), SB(_matIDsBuffer, 8), SB(_physMatsBuffer, 9),
			Img(_mpmParticlePositionTextureRid, 10), Img(_mpmParticleVelocityTextureRid, 11)
		});

		_densityUniformSet   = CreateUniformSet(_shaderDensity, new[]{
			UB(_paramsBuffer, 0), SB(_posMassBuffer, 1), SB(_densityBuffer, 2)
		});

		_marchingCubesUniformSet = CreateUniformSet(_shaderMC, new[]{
			UB(_paramsBuffer, 0), SB(_densityBuffer, 1), SB(_triTableBuffer, 2),
			Img(_mcPositionTextureRid, 3), Img(_mcNormalTextureRid, 4),
			SB(_mcCounterBuffer, 5)
		});
		
		_mcFinalizeUniformSet    = CreateUniformSet(_shaderMCFinalize, new[]{
			SB(_mcCounterBuffer, 0), Img(_mcVertexCountTextureRid, 1)
		});

		_constraintUniformSet = CreateUniformSet(_shaderConstraint, new[]{
			UB(_paramsBuffer, 0),
			SB(_deformationBuffer, 1),
			SB(_elasticFBuffer, 2),
			SB(_matIDsBuffer, 3),
			SB(_physMatsBuffer, 4)
		});
	}

	public override void _Process(double delta)
	{
		float frameDeltaTime = (float)delta * SimulationSpeed;
		int substeps = Mathf.CeilToInt(frameDeltaTime / (stableDt));
		substeps = Mathf.Max(substeps, 1);
		float actualDt = frameDeltaTime / substeps;
		_substepDt = actualDt;

		UpdateSimulationParams(GridDim, DomainSize / (float)GridDim, actualDt, new Vector3(0,0,0), (uint)_activeParticleCount);
		int idealSubsteps = substeps;
		substeps = Mathf.Min(substeps, MaxSubstepsPerFrame);

		float simulatedTime = substeps * actualDt;
		float actualSimSpeed = simulatedTime / (float)delta;
		
		
		//GD.Print($"[MPM] FPS: {1.0f / delta:F1} | Substeps: {substeps}/{idealSubsteps} | dt: {actualDt:F4}s | Sim Speed: {actualSimSpeed:F2}x");

		for (int i = 0; i < substeps; i++)
		{
			DispatchPhysicsSubstep();
			_impulsePending = false;
		}

		_meshUpdateTimer += Mathf.Max((float)delta, 0.5f);

		float meshUpdateInterval = 1.0f / MeshUpdateHz;

		if (_meshUpdateTimer >= meshUpdateInterval) {
			_meshUpdateTimer %= meshUpdateInterval;
			//DispatchMarchingCubes();
		}
	}

	public override void _UnhandledInput(InputEvent @event)
	{
		if (@event is not InputEventMouseButton mouseEvent
			|| mouseEvent.ButtonIndex != MouseButton.Left
			|| !mouseEvent.Pressed)
			return;

		Camera3D camera = GetViewport().GetCamera3D();
		if (camera == null)
			return;

		_impulseDirection = -camera.GlobalBasis.Z.Normalized();
		_impulsePosition = camera.GlobalPosition + _impulseDirection * ImpulseAimDistance;
		_impulsePending = true;
	}

	private void DispatchPhysicsSubstep()
	{
		for (int iteration = 0; iteration < PBIterations; iteration++)
		{
			// Bit 0 applies external acceleration once per physical substep;
			// bit 1 commits particle position/material state on the final solve pass.
			float iterationFlags = (iteration == 0 ? 1.0f : 0.0f)
				+ (iteration == PBIterations - 1 ? 2.0f : 0.0f);
			UpdateSimulationParams(GridDim, DomainSize / (float)GridDim,
				_substepDt,
				new Vector3(0, 0, 0), (uint)_activeParticleCount, iterationFlags,
				_impulsePending && iteration == 0 ? _impulsePosition : Vector3.Zero,
				_impulsePending && iteration == 0 ? _impulseDirection : Vector3.Zero,
				_impulsePending && iteration == 0 ? ImpulseRadius : 0.0f,
				_impulsePending && iteration == 0 ? ImpulseStrength : 0.0f);
			// -------------------------------------------------
			// Clear grid BEFORE beginning the compute list
			// -------------------------------------------------

			uint gridByteSize = (uint)(TotalGridCells * 16 * sizeof(float));
			_rd.BufferClear(_gridBuffer,0,gridByteSize);

			long computeList = _rd.ComputeListBegin();

			// -------------------------------------------------
			// PB-MPM constraint
			// -------------------------------------------------

			_rd.ComputeListBindComputePipeline(
				computeList,
				_constraintPipeline
			);

			_rd.ComputeListBindUniformSet(
				computeList,
				_constraintUniformSet,
				0
			);

			_rd.ComputeListDispatch(
				computeList,
				(uint)Mathf.CeilToInt(
					_activeParticleCount / 128.0f
				),
				1,
				1
			);
			_rd.ComputeListAddBarrier(computeList);

			// -------------------------------------------------
			// EXISTING P2G CODE
			// -------------------------------------------------

			_rd.ComputeListBindComputePipeline(computeList, _p2gPipeline);
			_rd.ComputeListBindUniformSet(computeList, _p2gUniformSet, 0);
			uint particleGroups = (uint)Mathf.CeilToInt((float)_activeParticleCount / 64.0f);
			_rd.ComputeListDispatch(computeList, particleGroups, 1, 1);
			_rd.ComputeListAddBarrier(computeList);

			// -------------------------------------------------
			// EXISTING GRID CODE
			// -------------------------------------------------

			_rd.ComputeListBindComputePipeline(computeList, _gridSolvePipeline);
			_rd.ComputeListBindUniformSet(computeList, _gridSolveUniformSet, 0);
			uint gridGroups = (uint)Mathf.CeilToInt((float)TotalGridCells / 64.0f);
			_rd.ComputeListDispatch(computeList, gridGroups, 1, 1);
			_rd.ComputeListAddBarrier(computeList);
			// -------------------------------------------------
			// EXISTING G2P CODE
			// -------------------------------------------------

			_rd.ComputeListBindComputePipeline(computeList, _g2pPipeline);
			_rd.ComputeListBindUniformSet(computeList, _g2pUniformSet, 0);
			_rd.ComputeListDispatch(computeList, particleGroups, 1, 1);
			_rd.ComputeListEnd();

			//_rd.ComputeListEnd();
		}
		_impulsePending = false;
	}

	private void DispatchMarchingCubes()
	{
		_rd.BufferClear(_densityBuffer, 0, (uint)(TotalGridCells * sizeof(int)));
		_rd.BufferClear(_mcCounterBuffer, 0, sizeof(uint));

		long computeList = _rd.ComputeListBegin();

		// -------------------------------------------------------------------------
		// Density
		// -------------------------------------------------------------------------
		_rd.ComputeListBindComputePipeline(computeList, _densityPipeline);
		_rd.ComputeListBindUniformSet(computeList, _densityUniformSet, 0);
		uint particleGroups = (uint)Mathf.CeilToInt((float)_activeParticleCount / 64.0f);
		_rd.ComputeListDispatch(computeList, particleGroups, 1, 1);
		_rd.ComputeListAddBarrier(computeList);

		// -------------------------------------------------------------------------
		// Marching cubes
		// -------------------------------------------------------------------------
		_rd.ComputeListBindComputePipeline(computeList, _marchingCubesPipeline);
		_rd.ComputeListBindUniformSet(computeList, _marchingCubesUniformSet, 0);
		uint groups = (uint)Mathf.CeilToInt(GridDim / 8.0f);
		_rd.ComputeListDispatch(computeList, groups, groups, groups);
		_rd.ComputeListAddBarrier(computeList);

		// -------------------------------------------------------------------------
		// Convert GPU buffers -> GPU textures.
		// -------------------------------------------------------------------------
		_rd.ComputeListBindComputePipeline(computeList,_mcFinalizePipeline);
		_rd.ComputeListBindUniformSet(computeList,_mcFinalizeUniformSet,0);
		_rd.ComputeListDispatch(computeList,1,1,1);

		// Make the generated image visible to the subsequent raster pass.
		_rd.ComputeListAddBarrier(computeList);

		_rd.ComputeListEnd();
	}

	// --- Helpers ---

	private Rid CreateMeshTexture(
		RenderingDevice.DataFormat dataFormat,
		uint width,
		uint height)
	{
		var format = new RDTextureFormat
		{
			Width = width,
			Height = height,
			Format = dataFormat,

			UsageBits =
				RenderingDevice.TextureUsageBits.StorageBit |
				RenderingDevice.TextureUsageBits.SamplingBit
		};

		return _rd.TextureCreate(
			format,
			new RDTextureView()
		);
	}

	private Rid CreateStorageBuffer(uint size, byte[] data = null) =>
		data == null ? _rd.StorageBufferCreate(size) : _rd.StorageBufferCreate(size, data);
	private Rid CreateUniformBuffer(uint size, byte[] data = null) =>
		data == null ? _rd.UniformBufferCreate(size) : _rd.UniformBufferCreate(size, data);

	private Rid CreateUniformSet(Rid shader, (Rid, RenderingDevice.UniformType, int)[] buffers)
	{
		var uniforms = new Godot.Collections.Array<RDUniform>();

		for (int i = 0; i < buffers.Length; i++) {
			var u = new RDUniform { UniformType = buffers[i].Item2, Binding = buffers[i].Item3 };
			u.AddId(buffers[i].Item1);
			uniforms.Add(u);
		}

		return _rd.UniformSetCreate(uniforms, shader, 0);
	}

	private static byte[] ToByteArray<T>(T[] array) where T : struct 
	{
		int size = Marshal.SizeOf<T>() * array.Length;
		byte[] bytes = new byte[size];
		GCHandle handle = GCHandle.Alloc(array, GCHandleType.Pinned);

		try { Marshal.Copy(handle.AddrOfPinnedObject(), bytes, 0, size); }
		finally { handle.Free(); }
		return bytes;
	}

	public static byte[] StructureToByteArray<T>(T structure) where T : struct
	{
		int size = Marshal.SizeOf(structure);
		byte[] array = new byte[size];
		IntPtr ptr = Marshal.AllocHGlobal(size);

		try { Marshal.StructureToPtr(structure, ptr, true); Marshal.Copy(ptr, array, 0, size); }
		finally { Marshal.FreeHGlobal(ptr); }
		return array;
	}

	public float DeriveStableDt(MaterialProps[] materials, float dx, float estimatedMaxVel)
	{
		float minAllowedDt = float.MaxValue;

		foreach (var mat in materials)
		{
			// 1. Calculate Lamé parameters & moduli
			float mu = mat.E / (2.0f * (1.0f + mat.NU));
			float lambda = (mat.E * mat.NU) / ((1.0f + mat.NU) * (1.0f - 2.0f * mat.NU));
			float K = lambda + (2.0f / 3.0f) * mu;

			// 2. Estimate wave speed c_s
			// (Assuming standard density rho_0 = 1000.0 kg/m^3 if not explicitly stored in mat)
			float rho0 = mat.density; 
			float c_s = Mathf.Sqrt((K + (4.0f / 3.0f) * mu) / rho0);

			// 3. Evaluate acoustic limit
			float dtWave = 0.3f * (dx / c_s);

			// 4. Evaluate kinematic limit
			float dtCFL = 0.5f * (dx / Mathf.Max(estimatedMaxVel, 1.0f));

			// 5. Track the tightest constraint across all active materials
			float matMaxDt = Mathf.Min(dtWave, dtCFL);
			if (matMaxDt < minAllowedDt) {
				minAllowedDt = matMaxDt;
			}
		}

		GD.Print(minAllowedDt);
		return minAllowedDt;
	}

	public void SetupShaderMaterial(ref MeshInstance3D mesh, ref Shader shader, ref ShaderMaterial shaderMat, string shaderFilename) {
		shader = GD.Load<Shader>(shaderFilename);
		if (shader == null) { throw new InvalidOperationException("Failed to load " + shaderFilename); }
		shaderMat = new ShaderMaterial { Shader = shader };
		mesh.MaterialOverride = shaderMat;
	}
}
