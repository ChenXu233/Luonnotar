import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';

/// Live camera preview rendered by the native pipeline (camera2 NDK ->
/// ANativeWindow). Uses TLHC (initSurfaceAndroidView) instead of the default
/// VirtualDisplay: with a 25fps SurfaceView, VD has to composite every layer
/// into the virtual display and blit it back, adding latency and jank; TLHC
/// puts the native Surface on screen directly with Flutter widgets composited
/// as an overlay on top.
class GlyphDetCameraView extends StatelessWidget {
  const GlyphDetCameraView({super.key});

  @override
  Widget build(BuildContext context) {
    return PlatformViewLink(
      viewType: 'glyph_det_camera_view',
      surfaceFactory: (context, controller) {
        return AndroidViewSurface(
          controller: controller as AndroidViewController,
          gestureRecognizers: const {},
          hitTestBehavior: PlatformViewHitTestBehavior.opaque,
        );
      },
      onCreatePlatformView: (params) {
        return PlatformViewsService.initSurfaceAndroidView(
          id: params.id,
          viewType: 'glyph_det_camera_view',
          layoutDirection: TextDirection.ltr,
          creationParamsCodec: const StandardMessageCodec(),
        )
          ..addOnPlatformViewCreatedListener(params.onPlatformViewCreated)
          ..create();
      },
    );
  }
}
