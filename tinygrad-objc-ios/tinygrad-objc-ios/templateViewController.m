#import "ViewController.h"
#import <Metal/Metal.h>

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    NSError *error = nil;
    id<MTLLibrary> library = [MTLCreateSystemDefaultDevice() newDefaultLibrary];
    CFTimeInterval startTime = CACurrentMediaTime();

    //CODE HERE
    //END
    //NSLog(@"%d", *(int32_t *)[x11bfeefc0_9 contents]);
    NSLog(@"Time: %f seconds", CACurrentMediaTime() - startTime);
}


@end
