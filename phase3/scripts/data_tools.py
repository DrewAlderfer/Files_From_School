def prnt_cls_rp(classification_report):
    """
    Describiton:
    Prints a classification report using Tabluate().
    Parameters:
        <str> (output from the sklearn.metrics.classification_report object)
    Returns:
        <str> tabulated string
    """
    if not isinstance(classification_report, str):
        raise TypeError("You have to pass the classification_report output as an arg.")
    report = classification_report
    report = report.split('\n')
    report = [x.split(" ") for x in report]
    for line in report:
        if len(line) == 1:
            report.remove(line)
    for line in report:
        count = 0
        for pos, itm in enumerate(line):
            if count > 12:
                line.insert(pos, "empty_cell")
                count = 0
                continue
            if itm != "":
                count = 0
            count += 1
        if "avg" in line:
            before = line.index('avg') - 1
            line[before] += " avg"
            line.remove("avg")
        while "" in line:
            line.remove("")
        while "empty_cell" in line:
            line[line.index("empty_cell")] = ""
    return print(tabulate(report, tablefmt="simple_grid"))
