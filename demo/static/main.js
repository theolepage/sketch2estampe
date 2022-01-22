const COLOR = "white"
const LINE_WIDTH = 2

var canvas, ctx, w, h
var flag = false, prevX = 0, currX = 0, prevY = 0, currY = 0, dot_flag = false

const init = () => {
    canvas = document.getElementById('canvas')
    ctx = canvas.getContext('2d')
    w = canvas.width
    h = canvas.height

    canvas.addEventListener('mousemove', e => update('move', e), false)
    canvas.addEventListener('mousedown', e => update('down', e), false)
    canvas.addEventListener('mouseup', e => update('up', e), false)
    canvas.addEventListener('mouseout', e => update('out', e), false)
}

const draw = () => {
    ctx.beginPath()
    ctx.moveTo(prevX, prevY)
    ctx.lineTo(currX, currY)
    ctx.strokeStyle = COLOR
    ctx.lineWidth = LINE_WIDTH
    ctx.stroke()
    ctx.closePath()
}

const erase = () => {
    ctx.clearRect(0, 0, w, h)
}

const update = (state, e) => {
    if (state == 'down') {
        prevX = currX
        prevY = currY
        currX = e.clientX - canvas.offsetLeft
        currY = e.clientY - canvas.offsetTop

        flag = true
        dot_flag = true
        if (dot_flag) {
            ctx.beginPath()
            ctx.fillStyle = COLOR
            ctx.fillRect(currX, currY, 2, 2)
            ctx.closePath()
            dot_flag = false
        }
    }

    if (state == 'up' || state == "out") {
        flag = false
    }

    if (state == 'move') {
        if (flag) {
            prevX = currX
            prevY = currY
            currX = e.clientX - canvas.offsetLeft
            currY = e.clientY - canvas.offsetTop
            draw()
        }
    }
}

const format_params_ajax_request = (data) => {
    const format_fn = key => {
        return encodeURIComponent(key) + '=' + encodeURIComponent(data[key])
    } 
    return Object.keys(data).map(format_fn).join('&')
}

const post_ajax_request = (url, data, success_callback) => {
    var xhr = new XMLHttpRequest()
    xhr.open('POST', url)
    xhr.onreadystatechange = () => {
        if (xhr.readyState > 3 && xhr.status == 200)
            success_callback(xhr.responseText)
    }
    xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest')
    xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded')
    xhr.send(format_params_ajax_request(data))
}

const render = () => {
    post_ajax_request(
        '/',
        {
            'image': canvas.toDataURL(),
            'checkpoint': document.getElementById('checkpoint').value
        },
        data => document.getElementById('result').setAttribute('src', data)
    )
}