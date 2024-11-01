from scipy.stats import chisquare, ks_2samp

def chiSquareTest(observed, expected, alpha=0.05):
    try:
        chiValue, pValue = chisquare(f_obs=observed, f_exp=expected)
        print(f"\n------------- Chi-Square test --------------")
        print(f"📊 Chi-Square value: {chiValue}")
        print(f"📉 P-Value: {pValue}")
        if pValue < alpha:
            print("❌ Reject the null hypothesis")
        else:
            print("✅ Fail to reject the null hypothesis")
    except ValueError as e:
        print(f"\n-------------------------- Chi-Square test ---------------------------")
        print("⚠️  The distributions do not match in size (or are not valid for the test).")
        print("⏭️  Skipping test...")

def kstest(observed, expected, alpha=0.05):
    kstestValue, pValue = ks_2samp(observed, expected)
    print(f"\n----------------- K-S Test -----------------")
    print(f"📊 K-S Test value: {kstestValue}")
    print(f"📉 P-Value: {pValue}")
    if pValue < alpha:
        print("❌ Reject the null hypothesis")
    else:
        print("✅ Fail to reject the null hypothesis")

if __name__ == "__main__":
    # Chi-Square Test
    # Accept the null hypothesis
    observed = [12, 15, 14, 10, 9, 11, 13, 8, 7, 16]
    expected = [12, 15, 14, 10, 9, 11, 13, 8, 7, 16] 
    chiSquareTest(observed, expected)

    # Reject the null hypothesis
    observed = [30, 5, 25, 10, 35, 40, 5, 10, 5, 35]
    expected = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]  
    chiSquareTest(observed, expected)

    # K-S Test
    # Accept the null hypothesis
    observed = [12, 15, 14, 10, 9, 11, 13, 8, 7, 16]
    expected = [12, 15, 14, 10, 9, 11, 13, 8, 7, 16] 
    kstest(observed, expected)

    # Reject the null hypothesis
    observed = [30, 5, 25, 10, 35, 40, 5, 10, 5, 35]
    expected = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]  
    kstest(observed, expected)