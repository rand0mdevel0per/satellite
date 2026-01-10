use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: llvm-config <flag>");
        process::exit(1);
    }

    let prefix = "D:\\LLVM";
    
    for arg in &args[1..] {
        match arg.as_str() {
            "--version" => {
                println!("21.1.7");
                return;
            }
            "--prefix" => {
                println!("{}", prefix);
                return;
            }
            "--bindir" => {
                println!("{}\\bin", prefix);
                return;
            }
            "--includedir" => {
                println!("{}\\include", prefix);
                return;
            }
            "--libdir" => {
                println!("{}\\lib", prefix);
                return;
            }
            "--cmakedir" => {
                println!("{}\\lib\\cmake\\llvm", prefix);
                return;
            }
            "--cxxflags" => {
                println!("-I{}\\include -D_DEBUG -D_MT -D_DLL --driver-mode=cl -DWIN32 -D_WINDOWS", prefix);
                return;
            }
            "--ldflags" => {
                println!("-L{}\\lib", prefix);
                return;
            }
            "--libs" => {
                // Return generic LLVM lib or list all?
                // Minimal output often works if using dynamic linking
                println!("-lLLVM"); 
                return;
            }
            "--system-libs" => {
                println!("");
                return;
            }
            "--link-static" => {
                // We want dynamic usually
                println!("false");
                return;
            }
             "--link-shared" => {
                println!("true");
                return;
            }
            _ => {}
        }
    }
    
    // Fallback for combo args like "--libs --system-libs"
    // Just handling simple case for now
}
