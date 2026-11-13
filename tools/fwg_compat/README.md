# Windows FWG filesystem compatibility

FWG CMIP6 Global v5.2.0 / Engine 1.1.0: GridCache.toRealPath can raise AccessDeniedException on readable Windows files. This sidecar replaces exactly one method invocation in GridCache.class with CanonicalPaths.resolve. It first attempts the original API, then uses File.getCanonicalFile only for AccessDeniedException on Windows, and only for a readable regular file. SHA256 fingerprinting and every scientific method remain unchanged. No ACLs or operating-system permissions are modified.

Original GridCache.class SHA256: `238b5233d700b802792d66be5db00ea421e538341bc98132fa3a7630be32b248`. Runtime refuses other versions. The official 3.7 GB distribution is unchanged; the sidecar precedes it on the classpath only on Windows. Cache identity includes the sidecar bytes.

The GridCache bytecode is derived from FWG by Eugénio Rodrigues, David Carvalho and Marco S. Fernandes, licensed CC BY-NC-SA 4.0. This derivative sidecar and its modifications are provided under the same license: https://creativecommons.org/licenses/by-nc-sa/4.0/ . Source: https://bitbucket.org/cling-project/futureweathergenerator/ . Scientific reference: https://doi.org/10.1016/j.buildenv.2023.110104 .

Build with JDK 21 (no runtime compiler required):

```
javac --add-exports java.base/jdk.internal.org.objectweb.asm=ALL-UNNAMED -d build tools/fwg_compat/BuildPatch.java tools/fwg_compat/CanonicalPaths.java
java --add-exports java.base/jdk.internal.org.objectweb.asm=ALL-UNNAMED -cp build BuildPatch vendor/fwg/FutureWeatherGenerator-CMIP6-Global-v5.2.0.jar build/climateclock/compat/CanonicalPaths.class assets/fwg-windows-compat.jar
```

BuildPatch uses the JDK's bundled bytecode visitor to preserve original method code and replace one invocation. It aborts unless exactly one lookup is found. The only added runtime class is CanonicalPaths. Expanded model CSVs use NTFS compression when available, reducing temporary disk requirements without changing file content.
