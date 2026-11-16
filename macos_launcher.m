#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    @autoreleasepool {
        const char *projectDir = "/Users/mima0000/.gemini/antigravity/scratch/voice-clone-realtime-translator";
        chdir(projectDir);

        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:@"data" withIntermediateDirectories:YES attributes:nil error:nil];

        int logFd = open("data/app.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (logFd >= 0) {
            dup2(logFd, STDOUT_FILENO);
            dup2(logFd, STDERR_FILENO);
            close(logFd);
        }

        // Trigger macOS TCC Microphone permission prompt & System Settings registration
        AVAuthorizationStatus status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio];
        if (status == AVAuthorizationStatusNotDetermined) {
            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio completionHandler:^(BOOL granted) {
                NSLog(@"[VoiceCloneLauncher] Microphone permission granted: %d", granted);
                dispatch_semaphore_signal(sem);
            }];
            dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(30 * NSEC_PER_SEC)));
        } else {
            NSLog(@"[VoiceCloneLauncher] Microphone authorization status: %ld", (long)status);
        }

        const char *pythonBin = "/Users/mima0000/.gemini/antigravity/scratch/voice-clone-realtime-translator/.venv/bin/python";
        char *const args[] = {
            (char *)pythonBin,
            "-m",
            "src.desktop_app",
            NULL
        };
        execv(pythonBin, args);
        perror("execv failed");
        return 1;
    }
}
