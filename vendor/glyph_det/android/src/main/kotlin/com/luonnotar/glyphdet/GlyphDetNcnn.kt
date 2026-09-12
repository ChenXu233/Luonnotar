package com.luonnotar.glyphdet

import android.view.Surface

class GlyphDetNcnn {
    external fun loadModel(paramPath: String, binPath: String, cpugpu: Int): Boolean
    external fun setQueryMask(mask: ByteArray): Boolean
    external fun openCamera(facing: Int): Boolean
    external fun closeCamera(): Boolean
    external fun setOutputWindow(surface: Surface): Boolean
    external fun clearOutputWindow(): Boolean
    external fun toggleFlash(): Boolean
    external fun pollResults(): String

    companion object {
        init {
            System.loadLibrary("glyphdet")
        }
    }
}
