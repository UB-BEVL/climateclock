import java.nio.file.*;
import java.util.zip.*;
import jdk.internal.org.objectweb.asm.*;
public class BuildPatch {
 public static void main(String[] a) throws Exception {
  String name="futureweathergenerator/io/GridCache.class";
  byte[] source;
  try(ZipFile zip=new ZipFile(a[0])) {source=zip.getInputStream(zip.getEntry(name)).readAllBytes();}
  ClassReader reader=new ClassReader(source); ClassWriter writer=new ClassWriter(0);
  int[] count={0};
  reader.accept(new ClassVisitor(Opcodes.ASM8,writer){
   public MethodVisitor visitMethod(int access,String n,String d,String sig,String[] ex){
    MethodVisitor mv=super.visitMethod(access,n,d,sig,ex);
    return new MethodVisitor(Opcodes.ASM8,mv){
     public void visitMethodInsn(int opcode,String owner,String method,String descriptor,boolean itf){
      if(owner.equals("java/nio/file/Path") && method.equals("toRealPath")) {
       count[0]++; super.visitMethodInsn(Opcodes.INVOKESTATIC,"climateclock/compat/CanonicalPaths","resolve","(Ljava/nio/file/Path;[Ljava/nio/file/LinkOption;)Ljava/nio/file/Path;",false);
      } else super.visitMethodInsn(opcode,owner,method,descriptor,itf);
     }
    };
   }
  },0);
  if(count[0]!=1)throw new IllegalStateException("Expected exactly one grid-path lookup, found "+count[0]);
  try(ZipOutputStream zip=new ZipOutputStream(Files.newOutputStream(Path.of(a[2])))){
   zip.putNextEntry(new ZipEntry(name));zip.write(writer.toByteArray());zip.closeEntry();
   zip.putNextEntry(new ZipEntry("climateclock/compat/CanonicalPaths.class"));zip.write(Files.readAllBytes(Path.of(a[1])));zip.closeEntry();
  }
  System.out.println("Patched exactly one path lookup; climate bytecode unchanged.");
 }
}
