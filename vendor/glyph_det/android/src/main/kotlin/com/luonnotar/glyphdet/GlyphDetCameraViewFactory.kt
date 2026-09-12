package com.luonnotar.glyphdet

import android.content.Context
import io.flutter.plugin.common.StandardMessageCodec
import io.flutter.plugin.platform.PlatformView
import io.flutter.plugin.platform.PlatformViewFactory

class GlyphDetCameraViewFactory(private val glyphDetNcnn: GlyphDetNcnn) : PlatformViewFactory(StandardMessageCodec.INSTANCE) {
    override fun create(context: Context, viewId: Int, args: Any?): PlatformView {
        val creationParams = args as Map<String?, Any?>?
        return GlyphDetCameraView(context, viewId, creationParams, glyphDetNcnn)
    }
}
