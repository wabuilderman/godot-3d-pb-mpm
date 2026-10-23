extends Label

var time_left: float = 0.0
const UPDATE_INTERVAL: float = 0.2 # Updates 5 times a second

func _process(delta: float) -> void:
	time_left -= delta
	if time_left <= 0:
		time_left = UPDATE_INTERVAL
		update_performance_stats(delta)

func update_performance_stats(delta: float) -> void:
	# 1. Frametime (in milliseconds) & FPS
	var fps: float = Engine.get_frames_per_second()
	var frametime_ms: float = delta_to_ms(delta)
	
	# 2. RAM Usage (Static memory allocated by Godot in MB)
	var ram_bytes: int = Performance.get_monitor(Performance.MEMORY_STATIC)
	var ram_mb: float = ram_bytes / (1024.0 * 1024.0)
	
	# 3. VRAM Usage (Video memory used by the renderer in MB)
	var vram_bytes: int = Performance.get_monitor(Performance.RENDER_VIDEO_MEM_USED)
	var vram_mb: float = vram_bytes / (1024.0 * 1024.0)
	
	# 4. Approximate process load time in ms
	var process_time_ms: float = Performance.get_monitor(Performance.TIME_PROCESS) * 1000.0

	# Format and display the text
	text = """
	FPS: %d (%3.2f ms)
	RAM (Static): %.2f MB
	VRAM Used: %.2f MB
	Process Time: %.2f ms
	""".strip_edges() % [fps, frametime_ms, ram_mb, vram_mb, process_time_ms]

func delta_to_ms(d: float) -> float:
	return d * 1000.0
