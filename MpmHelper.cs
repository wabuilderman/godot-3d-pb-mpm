using Godot;
using System;

public class MpmHelper {
	public static Rid CompileShader(RenderingDevice rd, string path)
	{
		if (rd == null)
		{
			GD.PrintErr("RenderingDevice is null! Ensure your project is set to Forward+ or Mobile rendering driver.");
			return new Rid();
		}

		if (!FileAccess.FileExists(path))
		{
			GD.PrintErr($"Shader file not found at path: {path}");
			return new Rid();
		}

		using var file = FileAccess.Open(path, FileAccess.ModeFlags.Read);
		if (file == null)
		{
			GD.PrintErr($"Failed to open shader file at path: {path}. Error: {FileAccess.GetOpenError()}");
			return new Rid();
		}

		string glslCode = file.GetAsText();

		var shaderSource = new RDShaderSource();
		shaderSource.SetStageSource(RenderingDevice.ShaderStage.Compute, glslCode);

		var spirv = rd.ShaderCompileSpirVFromSource(shaderSource);
		
		// Check if GLSL compilation yielded SPIR-V errors
		if (spirv.CompileErrorCompute.Length > 0)
		{
			GD.PrintErr($"GLSL Compilation Error in '{path}':\n{spirv.CompileErrorCompute}");
			return new Rid();
		}

		return rd.ShaderCreateFromSpirV(spirv);
	}
}
