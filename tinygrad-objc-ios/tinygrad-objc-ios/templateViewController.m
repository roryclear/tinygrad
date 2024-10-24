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

    //VARS
    //CODE
    //NSLog(@"%d", *((int32_t *)variable));
    NSLog(@"Time: %f seconds", CACurrentMediaTime() - startTime);
}


@end
