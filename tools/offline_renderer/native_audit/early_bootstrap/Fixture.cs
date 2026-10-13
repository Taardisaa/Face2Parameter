// Synthetic CPU entrypoint fixture only; never deploy this DLL to the game.
using System;
using System.IO;
namespace BepInEx.Preloader {
    public static class Entrypoint {
        public static void Main() {
            File.AppendAllText(Environment.GetEnvironmentVariable("HS2_BOOTSTRAP_CPU_LOG"),"original:"+Environment.GetEnvironmentVariable("DOORSTOP_INVOKE_DLL_PATH")+"\n");
        }
    }
}
