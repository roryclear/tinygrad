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
    //
        
    
    NSLog(@"Time: %f seconds", CACurrentMediaTime() - startTime);
}


@end
