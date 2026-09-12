plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

// fast_paddle_ocr 依赖的 uCrop 托管在 jitpack，需在本项目侧声明仓库
//（插件 build.gradle 里的 allprojects 只作用于插件子工程自身）
repositories {
    google()
    mavenCentral()
    maven("https://jitpack.io")
}

android {
    namespace = "com.luonnotar.luonnotar"
    compileSdk = flutter.compileSdkVersion
    // 与 fast_paddle_ocr / glyph_det 两个原生插件的 NDK 版本对齐，
    // 避免跨 NDK 的 STL 混链
    ndkVersion = "29.0.14206865"

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.luonnotar.luonnotar"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = 24
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName

        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    packaging {
        jniLibs {
            // fast_paddle_ocr 与 glyph_det 各自从同一 NDK 带出 libc++_shared.so，
            // 内容相同，任取一份避免重复打包冲突
            pickFirsts += listOf(
                "lib/arm64-v8a/libc++_shared.so",
                "lib/armeabi-v7a/libc++_shared.so",
            )
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}
