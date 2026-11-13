package climateclock.compat;
import java.nio.file.*;
import java.io.IOException;
public final class CanonicalPaths {
 public static Path resolve(Path path, LinkOption[] options) throws IOException {
  try { return path.toRealPath(options); }
  catch (AccessDeniedException original) {
   if (!System.getProperty("os.name").startsWith("Windows")) throw original;
   // No permissions are changed. The caller still reads and hashes this file.
   Path canonical = path.toFile().getCanonicalFile().toPath();
   if (!Files.isRegularFile(canonical) || !Files.isReadable(canonical)) throw original;
   return canonical;
  }
 }
}
