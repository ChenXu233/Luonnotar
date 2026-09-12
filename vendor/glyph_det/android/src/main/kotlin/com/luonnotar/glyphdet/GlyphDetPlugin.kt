package com.luonnotar.glyphdet

import android.content.Context
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result

/** GlyphDetPlugin */
class GlyphDetPlugin: FlutterPlugin, MethodCallHandler {
  private lateinit var channel : MethodChannel
  private lateinit var context: Context
  private val glyphDetNcnn = GlyphDetNcnn()
  private val inferExecutor = java.util.concurrent.Executors.newSingleThreadExecutor()

  override fun onAttachedToEngine(flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
    context = flutterPluginBinding.applicationContext
    channel = MethodChannel(flutterPluginBinding.binaryMessenger, "glyph_det")
    channel.setMethodCallHandler(this)

    flutterPluginBinding.platformViewRegistry.registerViewFactory(
        "glyph_det_camera_view", GlyphDetCameraViewFactory(glyphDetNcnn)
    )
  }

  override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
    channel.setMethodCallHandler(null)
    inferExecutor.shutdownNow()
  }

  override fun onMethodCall(call: MethodCall, result: Result) {
    when (call.method) {
        "getPlatformVersion" -> {
            result.success("Android ${android.os.Build.VERSION.RELEASE}")
        }
        "loadModel" -> {
            val paramPath = call.argument<String>("paramPath") ?: ""
            val binPath = call.argument<String>("binPath") ?: ""
            val cpugpu = call.argument<Int>("cpugpu") ?: 1
            // Vulkan first load compiles all compute pipelines on Adreno
            // (seconds); must leave the main thread or it ANRs
            inferExecutor.submit {
                try {
                    val success = glyphDetNcnn.loadModel(paramPath, binPath, cpugpu)
                    android.os.Handler(android.os.Looper.getMainLooper()).post {
                        result.success(success)
                    }
                } catch (e: Exception) {
                    android.os.Handler(android.os.Looper.getMainLooper()).post {
                        result.error("LOAD_ERROR", e.message, null)
                    }
                }
            }
        }
        "setQueryMask" -> {
            val mask = call.argument<ByteArray>("mask")
            if (mask == null) {
                result.error("INVALID_MASK", "mask bytes are required", null)
                return
            }
            val success = glyphDetNcnn.setQueryMask(mask)
            result.success(success)
        }
        "openCamera" -> {
            val facing = call.argument<Int>("facing") ?: 1
            val success = glyphDetNcnn.openCamera(facing)
            result.success(success)
        }
        "closeCamera" -> {
            val success = glyphDetNcnn.closeCamera()
            result.success(success)
        }
        "toggleFlash" -> {
            val success = glyphDetNcnn.toggleFlash()
            result.success(success)
        }
        "pollResults" -> {
            val json = glyphDetNcnn.pollResults()
            result.success(json)
        }
        else -> {
            result.notImplemented()
        }
    }
  }
}
