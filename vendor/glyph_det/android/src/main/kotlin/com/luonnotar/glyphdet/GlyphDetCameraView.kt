package com.luonnotar.glyphdet

import android.content.Context
import android.graphics.PixelFormat
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import io.flutter.plugin.platform.PlatformView

class GlyphDetCameraView(
    context: Context,
    id: Int,
    creationParams: Map<String?, Any?>?,
    private val glyphDetNcnn: GlyphDetNcnn
) : PlatformView, SurfaceHolder.Callback {

    private val surfaceView: SurfaceView = SurfaceView(context)

    init {
        surfaceView.holder.setFormat(PixelFormat.RGBA_8888)
        surfaceView.holder.addCallback(this)
    }

    override fun getView(): View {
        return surfaceView
    }

    override fun dispose() {
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        glyphDetNcnn.setOutputWindow(holder.surface)
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        glyphDetNcnn.clearOutputWindow()
    }
}
